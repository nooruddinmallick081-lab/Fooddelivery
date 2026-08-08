"""
inference.py — numpy-only forward pass for the ETA model.

Same philosophy as BeejBank: TensorFlow trains the model once, offline, in
model/train_model.ipynb. This file just replays the learned weights with
plain numpy (2 matmul+relu, 1 more matmul) so the shipped app never needs
to import TensorFlow. That one decision is why the portable build is a few
megabytes instead of half a gigabyte.

Four things this module does that the training notebook can't:

  1. **Guards the input domain.** The model only ever saw deliveries between
     roughly 1.5 km and 21 km. Ask it about a 90 km delivery and it will
     answer — confidently, and with nothing behind it. So we clamp to the
     range recorded in meta.json and *say that we clamped*.
  2. **Puts an error bar on every ETA.** We ship the model's own residual
     spread measured on held-out data (σ ≈ 4.5 min) and turn it into a real
     range. "27 minutes" alone is a bit of a lie; "27 min, usually within
     ±6" is a promise a dispatcher can actually plan against.
  3. **Explains a prediction deterministically.** See explain() — the first
     version of that function had the same bug BeejBank hit, and it took
     making the same mistake twice to properly learn the lesson.
  4. **Fails loudly on a schema mismatch.** If someone retrains with a
     different feature set and weights.npz drifts away from meta.json, you
     get an exception at import time instead of plausible-looking garbage
     forever.
"""

import json
import math
from pathlib import Path

import numpy as np

MODEL_DIR = Path(__file__).resolve().parent.parent / "model"

EARTH_RADIUS_KM = 6371.0

# meta.json layout this file knows how to read. Bump it here AND in
# model/train_model.ipynb together if the feature contract ever changes.
EXPECTED_SCHEMA_VERSION = 2

# z-multipliers for the confidence band. 80% is our default: a 95% band on a
# delivery ETA comes out so wide ("somewhere between 18 and 36 minutes") that
# it stops being useful to anyone waiting for food.
_Z_FOR_CONFIDENCE = {0.50: 0.674, 0.80: 1.282, 0.90: 1.645, 0.95: 1.960}

# A delivery that takes under 5 minutes isn't a delivery, it's a walk to the
# counter. Floor the band there rather than letting the maths print "2 min".
MIN_PLAUSIBLE_ETA_MIN = 5.0

# what to call each feature when a human has to read it off a chart
FEATURE_LABELS = {
    "distance_km": "Distance",
    "traffic_ordinal": "Traffic",
    "multiple_deliveries": "Rider's other orders",
    "Vehicle_condition": "Vehicle condition",
    "Delivery_person_Ratings": "Rider rating",
    "Delivery_person_Age": "Rider age",
    "festival_flag": "Festival day",
    "city": "Area type",
    "weather": "Weather",
    "order": "Order type",
    "vehicle": "Vehicle type",
}


class ModelSchemaError(RuntimeError):
    """weights.npz and meta.json don't agree, or meta.json is from an older build."""


def haversine_km(lat1, lon1, lat2, lon2):
    """Great-circle distance in km between two GPS points.

    Note the abs() on the first line — that IS the bug fix from the notebook,
    not a stray defensive habit. Some of this dataset's coordinates carry a
    sign error that puts Indian restaurants in the southern hemisphere, which
    made distance look completely uncorrelated with delivery time until we
    caught it. Training uses the same abs(), so inference has to as well or
    the two would quietly disagree about what "distance" means.
    """
    lat1, lon1, lat2, lon2 = (abs(v) for v in (lat1, lon1, lat2, lon2))
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return float(2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(a)))


class ETAModel:
    """Loads weights.npz + meta.json and predicts delivery time (minutes)."""

    def __init__(self, model_dir: Path = MODEL_DIR):
        model_dir = Path(model_dir)
        weights_path = model_dir / "weights.npz"
        meta_path = model_dir / "meta.json"
        for p in (weights_path, meta_path):
            if not p.exists():
                raise FileNotFoundError(
                    f"{p.name} is missing from {model_dir}. Retrain with "
                    f"model/train_model.ipynb, or restore it from the repo."
                )

        with np.load(weights_path) as weights:
            self.W1, self.b1 = weights["W1"], weights["b1"]
            self.W2, self.b2 = weights["W2"], weights["b2"]
            self.W3, self.b3 = weights["W3"], weights["b3"]

        with open(meta_path, encoding="utf-8") as f:
            self.meta = json.load(f)

        self._check_schema()

        self.feature_order: list[str] = self.meta["feature_order"]
        self.mean = np.asarray(self.meta["scaler_mean"], dtype="float32")
        self.std = np.asarray(self.meta["scaler_std"], dtype="float32")
        self.weather_categories: list[str] = self.meta["weather_categories"]
        self.order_categories: list[str] = self.meta["order_categories"]
        self.vehicle_categories: list[str] = self.meta["vehicle_categories"]
        self.city_categories: list[str] = self.meta["city_categories"]
        self.traffic_order: dict = self.meta["traffic_order"]  # {"Low":0 ... "Jam":3}
        self.traffic_levels: list[str] = list(self.traffic_order.keys())

        self.input_domain = self.meta["input_domain"]
        self.metrics = self.meta.get("metrics", {})
        self._sigma = float(self.meta["residual_sigma_minutes"])

        # Column indices for each one-hot block, so explain() can treat e.g.
        # the seven weather columns as ONE categorical feature instead of
        # seven unrelated numbers.
        self._onehot_blocks = {
            "city": {c: self.feature_order.index(f"city_{c}") for c in self.city_categories},
            "weather": {c: self.feature_order.index(f"weather_{c}") for c in self.weather_categories},
            "order": {c: self.feature_order.index(f"order_{c}") for c in self.order_categories},
            "vehicle": {c: self.feature_order.index(f"vehicle_{c}") for c in self.vehicle_categories},
        }
        # genuinely continuous inputs — the only ones a ±1σ nudge makes sense on
        self._continuous_features = [
            "distance_km", "multiple_deliveries", "Vehicle_condition",
            "Delivery_person_Ratings", "Delivery_person_Age",
        ]

    def _check_schema(self) -> None:
        version = self.meta.get("schema_version")
        if version != EXPECTED_SCHEMA_VERSION:
            raise ModelSchemaError(
                f"meta.json is schema v{version}, this build of inference.py expects "
                f"v{EXPECTED_SCHEMA_VERSION}. Re-run model/train_model.ipynb to "
                f"regenerate weights.npz + meta.json together."
            )
        n_features = len(self.meta["feature_order"])
        if self.W1.shape[0] != n_features:
            raise ModelSchemaError(
                f"weights.npz expects {self.W1.shape[0]} input features but meta.json "
                f"lists {n_features}. These two files are out of sync — retrain."
            )
        for name in ("scaler_mean", "scaler_std"):
            if len(self.meta[name]) != n_features:
                raise ModelSchemaError(
                    f"meta.json '{name}' has {len(self.meta[name])} entries, "
                    f"expected {n_features}."
                )

    # -- input guarding -------------------------------------------------------

    def clamp_inputs(self, distance_km, multiple_deliveries, vehicle_condition,
                     rider_rating, rider_age) -> tuple[dict, list[str]]:
        """Pull inputs back into the range the model was actually trained on.

        Returns (clamped_values, warnings). We clamp rather than refuse — a
        dispatcher with a genuinely unusual 40 km run still deserves a number,
        they just also deserve to be told we're answering from the edge of what
        we've seen rather than the middle of it.

        The distance bound is the one that matters in practice. Every delivery
        in the training set falls between about 1.5 km and 21 km, so the app's
        old "type any number you like" box could put the model 10x outside its
        own experience without a word of complaint.
        """
        raw = {
            "distance_km": float(distance_km),
            "multiple_deliveries": float(multiple_deliveries),
            "vehicle_condition": float(vehicle_condition),
            "rider_rating": float(rider_rating),
            "rider_age": float(rider_age),
        }
        pretty = {
            "distance_km": ("Distance", " km"),
            "multiple_deliveries": ("Concurrent orders", ""),
            "vehicle_condition": ("Vehicle condition", ""),
            "rider_rating": ("Rider rating", "★"),
            "rider_age": ("Rider age", ""),
        }

        clamped, warnings = {}, []
        for key, value in raw.items():
            if not math.isfinite(value):
                raise ValueError(f"{pretty[key][0]} must be a finite number, got {value!r}")
            bounds = self.input_domain[key]
            lo, hi = float(bounds["min"]), float(bounds["max"])
            fixed = min(max(value, lo), hi)
            clamped[key] = fixed
            if fixed != value:
                label, unit = pretty[key]
                warnings.append(
                    f"{label} {value:g}{unit} is outside the model's training range "
                    f"({lo:g}–{hi:g}{unit}); using {fixed:g}{unit} instead."
                )
        return clamped, warnings

    # -- feature building -----------------------------------------------------

    def _build_feature_vector(self, distance_km, traffic_level, multiple_deliveries,
                              vehicle_condition, rider_rating, rider_age, festival,
                              city, weather, order_type, vehicle_type) -> np.ndarray:
        row = {
            "distance_km": distance_km,
            "traffic_ordinal": self.traffic_order.get(traffic_level, 1),
            "multiple_deliveries": multiple_deliveries,
            "Vehicle_condition": vehicle_condition,
            "Delivery_person_Ratings": rider_rating,
            "Delivery_person_Age": rider_age,
            "festival_flag": 1.0 if festival else 0.0,
        }
        for c in self.city_categories:
            row[f"city_{c}"] = 1.0 if c == city else 0.0
        for c in self.weather_categories:
            row[f"weather_{c}"] = 1.0 if c == weather else 0.0
        for c in self.order_categories:
            row[f"order_{c}"] = 1.0 if c == order_type else 0.0
        for c in self.vehicle_categories:
            row[f"vehicle_{c}"] = 1.0 if c == vehicle_type else 0.0

        return np.array([row[name] for name in self.feature_order], dtype="float32")

    def _scale(self, x: np.ndarray) -> np.ndarray:
        return (x - self.mean) / self.std

    # -- forward pass ---------------------------------------------------------

    def _forward(self, x_scaled: np.ndarray) -> np.ndarray:
        """Replays the trained network. Works on one row or a whole batch.

        This is the entire "AI" at runtime: two matmuls each followed by a
        relu, then one more matmul. That's what a 3-layer MLP *is* — the
        500MB of TensorFlow is for training it, not for running it.
        """
        x = np.atleast_2d(x_scaled)
        h1 = np.maximum(0, x @ self.W1 + self.b1)   # dense_1 + relu
        h2 = np.maximum(0, h1 @ self.W2 + self.b2)  # dense_2 + relu
        return (h2 @ self.W3 + self.b3).ravel()     # linear output, minutes

    # -- public prediction API ------------------------------------------------

    def predict_eta(self, distance_km, traffic_level, multiple_deliveries,
                    vehicle_condition, rider_rating, rider_age, festival,
                    city, weather, order_type, vehicle_type) -> float:
        """Predicted delivery time in minutes (point estimate only).

        Kept as the simple one-number entry point because plenty of callers —
        the what-if curve, the marketplace, the tests — genuinely only want
        the number. Anything user-facing should prefer predict_with_interval()
        so the range and any clamp warnings travel with it.
        """
        return self.predict_with_interval(
            distance_km, traffic_level, multiple_deliveries, vehicle_condition,
            rider_rating, rider_age, festival, city, weather, order_type, vehicle_type,
        )["eta_min"]

    def predict_with_interval(self, distance_km, traffic_level, multiple_deliveries,
                              vehicle_condition, rider_rating, rider_age, festival,
                              city, weather, order_type, vehicle_type,
                              confidence: float = 0.80) -> dict:
        """ETA plus an honest range around it.

        The band width comes from this model's own residual spread on held-out
        validation data (σ ≈ 4.5 minutes), not from a number we picked because
        it looked reassuring. It's symmetric because — unlike BeejBank, which
        predicts in log space — this model predicts minutes directly, so the
        errors really are roughly symmetric. We only floor the lower edge, at
        MIN_PLAUSIBLE_ETA_MIN, because "we'll be there in 2 minutes" isn't a
        delivery promise anyone should print.
        """
        if confidence not in _Z_FOR_CONFIDENCE:
            raise ValueError(
                f"confidence must be one of {sorted(_Z_FOR_CONFIDENCE)}, got {confidence}"
            )
        clamped, warnings = self.clamp_inputs(
            distance_km, multiple_deliveries, vehicle_condition, rider_rating, rider_age
        )
        if traffic_level not in self.traffic_order:
            warnings.append(
                f"Unknown traffic level {traffic_level!r}; assuming Medium."
            )
        x = self._build_feature_vector(
            clamped["distance_km"], traffic_level, clamped["multiple_deliveries"],
            clamped["vehicle_condition"], clamped["rider_rating"], clamped["rider_age"],
            festival, city, weather, order_type, vehicle_type,
        )
        raw = float(self._forward(self._scale(x))[0])
        point = max(MIN_PLAUSIBLE_ETA_MIN, raw)

        z = _Z_FOR_CONFIDENCE[confidence]
        half_width = z * self._sigma
        return {
            "eta_min": point,
            "low_min": max(MIN_PLAUSIBLE_ETA_MIN, point - half_width),
            "high_min": point + half_width,
            "confidence": confidence,
            "sigma_minutes": self._sigma,
            "warnings": warnings,
            "clamped_inputs": clamped,
            "traffic_level": traffic_level,
        }

    def predict_many(self, distance_series, traffic_level, multiple_deliveries,
                     vehicle_condition, rider_rating, rider_age, festival,
                     city, weather, order_type, vehicle_type) -> np.ndarray:
        """Vectorised sweep over a range of distances.

        The what-if simulator redraws its whole curve on every slider drag. The
        first version looped 30 separate predictions in Python to do that; this
        builds one batch and pushes it through the network in a single matmul.
        Identical numbers — there's a test that asserts it matches the loop
        exactly — just not paying the per-call overhead 30 times while
        somebody is dragging.
        """
        distances = np.asarray(distance_series, dtype="float64").ravel()
        if distances.size == 0:
            return np.array([], dtype="float64")

        bounds = self.input_domain["distance_km"]
        distances = np.clip(distances, float(bounds["min"]), float(bounds["max"]))

        # everything except distance is identical down the sweep, so build one
        # row and overwrite just that column
        clamped, _ = self.clamp_inputs(
            distances[0], multiple_deliveries, vehicle_condition, rider_rating, rider_age
        )
        template = self._build_feature_vector(
            clamped["distance_km"], traffic_level, clamped["multiple_deliveries"],
            clamped["vehicle_condition"], clamped["rider_rating"], clamped["rider_age"],
            festival, city, weather, order_type, vehicle_type,
        )
        batch = np.tile(template, (distances.size, 1))
        batch[:, self.feature_order.index("distance_km")] = distances

        return np.maximum(MIN_PLAUSIBLE_ETA_MIN, self._forward(self._scale(batch)))

    # -- explainability -------------------------------------------------------

    def explain(self, distance_km, traffic_level, multiple_deliveries,
                vehicle_condition, rider_rating, rider_age, festival,
                city, weather, order_type, vehicle_type) -> dict[str, float]:
        """Local sensitivity: which inputs actually moved *this* prediction?

        For each genuinely continuous input we shift it by ±1 standard
        deviation (in scaled space, so literally ±1.0), holding everything
        else still, and record how far the ETA travels. Everything else here
        is a category, so we ask the proper counterfactual instead — "what
        would this same order look like in every other kind of weather?" —
        and average the swing.

        The first version of this nudged every column with a random gaussian
        draw. Two things were wrong with that, and BeejBank had already taught
        us both: the answer changed on every refresh, and the eighteen one-hot
        columns were getting continuous jitter, which put them in states like
        "0.6 Foggy" that the model has never seen and cannot reason about — so
        they scored high purely for being pushed somewhere nonsensical. It's
        fully deterministic now, and a category is treated as a category.

        Returns {human-readable feature name: % of total influence}.
        """
        clamped, _ = self.clamp_inputs(
            distance_km, multiple_deliveries, vehicle_condition, rider_rating, rider_age
        )
        base_x = self._build_feature_vector(
            clamped["distance_km"], traffic_level, clamped["multiple_deliveries"],
            clamped["vehicle_condition"], clamped["rider_rating"], clamped["rider_age"],
            festival, city, weather, order_type, vehicle_type,
        )
        base_scaled = self._scale(base_x)
        base_pred = float(self._forward(base_scaled)[0])

        sensitivities: dict[str, float] = {}

        # 1. continuous inputs — symmetric ±1σ probe, averaged
        probes = []
        for name in self._continuous_features:
            i = self.feature_order.index(name)
            for step in (+1.0, -1.0):
                nudged = base_scaled.copy()
                nudged[i] += step
                probes.append(nudged)
        swings = np.abs(self._forward(np.vstack(probes)) - base_pred)
        for j, name in enumerate(self._continuous_features):
            sensitivities[FEATURE_LABELS[name]] = float(swings[2 * j: 2 * j + 2].mean())

        # 2. traffic — ordinal, but the user picks it from a list of four, so
        #    the honest question is "what if this run were in the other three?"
        traffic_alts = [t for t in self.traffic_levels if t != traffic_level]
        if traffic_alts:
            cfs = []
            slot = self.feature_order.index("traffic_ordinal")
            for alt in traffic_alts:
                swapped = base_x.copy()
                swapped[slot] = self.traffic_order[alt]
                cfs.append(self._scale(swapped))
            sensitivities[FEATURE_LABELS["traffic_ordinal"]] = float(
                np.abs(self._forward(np.vstack(cfs)) - base_pred).mean()
            )

        # 3. festival — binary, so there is exactly one counterfactual: flip it
        slot = self.feature_order.index("festival_flag")
        flipped = base_x.copy()
        flipped[slot] = 0.0 if festival else 1.0
        sensitivities[FEATURE_LABELS["festival_flag"]] = float(
            abs(self._forward(self._scale(flipped))[0] - base_pred)
        )

        # 4. the four one-hot blocks — swap the whole block to each alternative
        current = {"city": city, "weather": weather,
                   "order": order_type, "vehicle": vehicle_type}
        for block, slots in self._onehot_blocks.items():
            alts = [c for c in slots if c != current[block]]
            if not alts:
                continue
            cfs = []
            for alt in alts:
                swapped = base_x.copy()
                for cat, idx in slots.items():
                    swapped[idx] = 1.0 if cat == alt else 0.0
                cfs.append(self._scale(swapped))
            sensitivities[FEATURE_LABELS[block]] = float(
                np.abs(self._forward(np.vstack(cfs)) - base_pred).mean()
            )

        total = sum(sensitivities.values())
        if total <= 0:
            # every probe landed in the same dead relu region — vanishingly
            # rare, but say "we don't know" rather than divide by zero
            return {name: 0.0 for name in sensitivities}
        return {
            name: round(100 * value / total, 1)
            for name, value in sorted(sensitivities.items(), key=lambda kv: -kv[1])
        }

    # -- model card -----------------------------------------------------------

    def model_card(self) -> dict:
        """The 'how much should you trust this' summary, for the UI to show.

        Every number in here was measured on held-out validation data by the
        training notebook — none of it is hardcoded or aspirational.
        """
        trained = self.meta.get("trained_with", {})
        return {
            "schema_version": self.meta["schema_version"],
            "n_features": len(self.feature_order),
            "architecture": f"{len(self.feature_order)} → {self.W1.shape[1]} → "
                            f"{self.W2.shape[1]} → {self.W3.shape[1]}",
            "n_training_rows": trained.get("n_rows"),
            "n_validation_rows": trained.get("n_validation_rows"),
            "mae_minutes": self.metrics.get("val_mae_minutes"),
            "rmse_minutes": self.metrics.get("val_rmse_minutes"),
            "r2": self.metrics.get("val_r2"),
            "medape_pct": self.metrics.get("val_medape_pct"),
            "pct_within_5_min": self.metrics.get("pct_within_5_min"),
            "pct_within_10_min": self.metrics.get("pct_within_10_min"),
            "per_traffic": self.metrics.get("per_traffic", {}),
            "per_city": self.metrics.get("per_city", {}),
            "permutation_importance": self.meta.get("permutation_importance", {}),
            "input_domain": self.input_domain,
            "sigma_minutes": self._sigma,
        }

    def accuracy_for_traffic(self, traffic_level: str) -> dict | None:
        """Validation accuracy for one traffic level, or None if too few rows.

        "In Jam traffic we're typically within 4 minutes" tells a dispatcher
        something they can act on. One blended average across every condition
        tells them almost nothing, and quietly flatters us in Low traffic.
        """
        return self.metrics.get("per_traffic", {}).get(traffic_level)
