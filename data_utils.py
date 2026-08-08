"""
data_utils.py — loading the delivery dataset + the map/analytics logic.

Unlike BeejBank (which needed a separate geojson boundary file to draw
India's states), this dataset already has real lat/long points for every
single order — restaurant location AND delivery location. So instead of a
state-level choropleth, we can plot actual delivery hotspots directly.

Two things we found in this file that are worth knowing about:

  THE NULL ISLAND BUG. 3,640 of the 45,584 rows have their coordinates
  recorded as exactly 0.0 — not a real place, just a missing value that
  someone wrote as a zero. Taking abs() of the coordinates (which we have
  to do, see the notebook's sign-error story) doesn't help, because zero
  is already positive. So all 3,640 landed in one grid cell at (0, 0),
  which is a spot in the Gulf of Guinea off West Africa, and it showed up
  as the second-biggest "delivery hotspot" on our map. It looked
  completely plausible if you didn't check where the dot was. We drop
  those rows from anything geographic now.

  THE CITY CODES. Every Delivery_person_ID looks like "JAPRES09DEL03",
  and the bit before "RES" is a real city code — JAP, HYD, CHEN, MUM and
  19 more. We only noticed because the IDs looked oddly structured. We
  checked it by taking the median coordinate of every code and seeing
  whether it landed on the city we'd guessed: JAP came out at 26.91N
  75.79E, which is Jaipur, and all 22 checked out the same way. So the map
  can name actual Indian cities instead of labelling zones "grid cell
  26.9, 75.8", and none of that required data we didn't already have.
"""

from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "food_delivery.csv"

GRID_SIZE_DEG = 0.5  # ~55km grid cells at the equator — coarse on purpose,
                     # this dataset's coordinates are semi-synthetic (see the
                     # notebook), fine-grained clustering would be false precision

# Below this, a coordinate isn't a location, it's a missing value someone
# recorded as a zero. Real Indian delivery coordinates are all well above 8°N.
NULL_ISLAND_TOLERANCE_DEG = 1.0

# Delivery_person_ID prefix -> the city it actually refers to. Confirmed by
# taking each code's median restaurant coordinate and checking it lands on the
# city we expected — every one of the 22 did.
CITY_CODES = {
    "AGR": "Agra", "ALH": "Prayagraj", "AURG": "Aurangabad", "BANG": "Bengaluru",
    "BHP": "Bhopal", "CHEN": "Chennai", "COIMB": "Coimbatore", "DEH": "Dehradun",
    "GOA": "Goa", "HYD": "Hyderabad", "INDO": "Indore", "JAP": "Jaipur",
    "KNP": "Kanpur", "KOC": "Kochi", "KOL": "Kolkata", "LUDH": "Ludhiana",
    "MUM": "Mumbai", "MYS": "Mysuru", "PUNE": "Pune", "RANCHI": "Ranchi",
    "SUR": "Surat", "VAD": "Vadodara",
}

COORD_COLS = [
    "Restaurant_latitude", "Restaurant_longitude",
    "Delivery_location_latitude", "Delivery_location_longitude",
]


@lru_cache(maxsize=1)
def _load_cached() -> pd.DataFrame:
    """The one real read of the CSV. Everything else works off a copy of this."""
    df = pd.read_csv(DATA_PATH)

    # same GPS fix as training — see model/train_model.ipynb for the story
    for col in COORD_COLS:
        df[col] = df[col].abs()

    # Same fills the training notebook uses, in the same order. These have to
    # agree: the model was trained on median-filled ages, so if this loader
    # left them as NaN, anything feeding a dataset row back through the model
    # would blow up on a null the model never saw. (It did — a test caught it.)
    df["Delivery_person_Age"] = df["Delivery_person_Age"].fillna(
        df["Delivery_person_Age"].median()
    )
    df["Delivery_person_Ratings"] = df["Delivery_person_Ratings"].fillna(
        df["Delivery_person_Ratings"].median()
    )
    df["multiple_deliveries"] = df["multiple_deliveries"].fillna(0)
    df["Road_traffic_density"] = df["Road_traffic_density"].fillna("Medium")
    df["Weather_conditions"] = df["Weather_conditions"].fillna("Unknown")
    df["City"] = df["City"].fillna("Metropolitian")
    df["Festival"] = df["Festival"].fillna("No")

    # the city hiding inside the rider ID — "JAPRES09DEL03" -> "JAP" -> Jaipur
    code = df["Delivery_person_ID"].astype("string").str.split("RES").str[0]
    df["city_code"] = code
    df["city_name"] = code.map(CITY_CODES).fillna("Unknown")

    # hour the order was placed. Times are "HH:MM" and 1,731 rows are blank,
    # so errors="coerce" leaves those as NaT rather than killing the load.
    ordered = pd.to_datetime(df["Time_Orderd"], format="%H:%M", errors="coerce")
    df["order_hour"] = ordered.dt.hour

    # true only where we have a coordinate that means something
    df["has_real_coords"] = (
        df[COORD_COLS].abs() > NULL_ISLAND_TOLERANCE_DEG
    ).all(axis=1)

    return df


def load_delivery_data() -> pd.DataFrame:
    """The cleaned dataset.

    Hands out a COPY every time, deliberately. Several screens read this same
    table and at least one of them adds a column; without the copy that column
    leaks into every other screen's view of the data. BeejBank got bitten by
    exactly this and it cost an evening to track down — copying 45k rows costs
    a few milliseconds.
    """
    return _load_cached().copy()


def dataset_summary() -> dict:
    """Headline facts about the data, for the UI to show instead of hardcoding.

    If we ever swap the CSV, these numbers follow it automatically rather than
    quietly becoming a lie in the README and on three different screens.
    """
    df = _load_cached()
    return {
        "n_orders": int(len(df)),
        "n_riders": int(df["Delivery_person_ID"].nunique()),
        "n_cities": int(df.loc[df["city_name"] != "Unknown", "city_name"].nunique()),
        "n_geo_usable": int(df["has_real_coords"].sum()),
        "n_null_island": int((~df["has_real_coords"]).sum()),
        "mean_delivery_min": round(float(df["Time_taken (min)"].mean()), 1),
        "fastest_min": int(df["Time_taken (min)"].min()),
        "slowest_min": int(df["Time_taken (min)"].max()),
        "festival_share_pct": round(float((df["Festival"] == "Yes").mean() * 100), 1),
    }


def delay_hotspots(df: pd.DataFrame | None = None, min_orders: int = 15) -> pd.DataFrame:
    """
    Buckets restaurant locations into a coarse lat/long grid and computes the
    average delivery time + order count per cell, now labelled with the real
    city each cell belongs to.

    Two filters keep this map honest:
      - rows without usable coordinates are dropped, so the 3,640 orders
        recorded at (0, 0) stop appearing as a giant phantom hotspot in the
        Atlantic (see this module's docstring)
      - cells with fewer than `min_orders` are dropped, because a "hotspot"
        computed from two deliveries isn't a pattern, it's noise
    """
    df = df if df is not None else load_delivery_data()
    geo = df[df["has_real_coords"]]
    if geo.empty:
        return pd.DataFrame(
            columns=["grid_lat", "grid_lon", "avg_delay", "order_count", "city"]
        )

    grid_lat = (geo["Restaurant_latitude"] / GRID_SIZE_DEG).round() * GRID_SIZE_DEG
    grid_lon = (geo["Restaurant_longitude"] / GRID_SIZE_DEG).round() * GRID_SIZE_DEG

    grouped = pd.DataFrame({
        "grid_lat": grid_lat,
        "grid_lon": grid_lon,
        "Time_taken": geo["Time_taken (min)"],
        "city": geo["city_name"],
    }).groupby(["grid_lat", "grid_lon"]).agg(
        avg_delay=("Time_taken", "mean"),
        order_count=("Time_taken", "size"),
        # a grid cell can straddle two cities; name it after whichever one
        # contributes most of the orders in it
        city=("city", lambda s: s.mode().iat[0] if not s.mode().empty else "Unknown"),
    ).reset_index()

    grouped = grouped[grouped["order_count"] >= min_orders].copy()
    grouped["avg_delay"] = grouped["avg_delay"].round(1)
    return grouped.sort_values("avg_delay", ascending=False).reset_index(drop=True)


def city_performance(df: pd.DataFrame | None = None, min_orders: int = 100) -> pd.DataFrame:
    """Average delivery time per real city, with the traffic mix behind it.

    The grid map answers "where", this answers "which city, and why" — a city
    can be slow because it's congested or because its orders travel further,
    and those need different responses from whoever plans rider coverage.
    """
    df = df if df is not None else load_delivery_data()
    known = df[df["city_name"] != "Unknown"]
    if known.empty:
        return pd.DataFrame(
            columns=["city", "avg_delay", "order_count", "jam_share_pct", "avg_rider_rating"]
        )

    out = known.groupby("city_name").agg(
        avg_delay=("Time_taken (min)", "mean"),
        order_count=("Time_taken (min)", "size"),
        jam_share_pct=("Road_traffic_density", lambda s: (s == "Jam").mean() * 100),
        avg_rider_rating=("Delivery_person_Ratings", "mean"),
    ).reset_index().rename(columns={"city_name": "city"})

    out = out[out["order_count"] >= min_orders].copy()
    for col in ("avg_delay", "jam_share_pct", "avg_rider_rating"):
        out[col] = out[col].round(2)
    return out.sort_values("avg_delay", ascending=False).reset_index(drop=True)


def rush_hour_profile(df: pd.DataFrame | None = None, min_orders: int = 50) -> pd.DataFrame:
    """Average delivery time by hour of day.

    This one genuinely surprised us. The gap between the quietest hour and the
    dinner rush is about 12 minutes — bigger than the effect of most things the
    model DOES take as an input. It isn't a model feature because the training
    notebook never built it, which is a fair criticism of the model and an
    honest thing to show rather than hide: the app can still tell a restaurant
    "your 8pm orders run 12 minutes longer than your 9am ones" straight from
    the data, no prediction required.
    """
    df = df if df is not None else load_delivery_data()
    timed = df[df["order_hour"].notna()]
    if timed.empty:
        return pd.DataFrame(columns=["hour", "avg_delay", "order_count", "label"])

    out = timed.groupby("order_hour").agg(
        avg_delay=("Time_taken (min)", "mean"),
        order_count=("Time_taken (min)", "size"),
    ).reset_index().rename(columns={"order_hour": "hour"})

    out = out[out["order_count"] >= min_orders].copy()
    out["hour"] = out["hour"].astype(int)
    out["avg_delay"] = out["avg_delay"].round(1)
    out["label"] = out["hour"].map(lambda h: f"{h % 12 or 12}{'am' if h < 12 else 'pm'}")
    return out.sort_values("hour").reset_index(drop=True)


def condition_impact(df: pd.DataFrame | None = None) -> dict[str, pd.DataFrame]:
    """How much each condition actually shifts delivery time, straight from the data.

    Deliberately NOT the model's opinion — this is the raw historical average
    per category. Having both means someone can check the model against the
    data it learned from instead of taking our word for it.
    """
    df = df if df is not None else load_delivery_data()
    baseline = float(df["Time_taken (min)"].mean())

    def by(column: str, label: str) -> pd.DataFrame:
        out = df.groupby(column).agg(
            avg_delay=("Time_taken (min)", "mean"),
            order_count=("Time_taken (min)", "size"),
        ).reset_index().rename(columns={column: label})
        out["vs_average_min"] = (out["avg_delay"] - baseline).round(1)
        out["avg_delay"] = out["avg_delay"].round(1)
        return out.sort_values("avg_delay", ascending=False).reset_index(drop=True)

    return {
        "traffic": by("Road_traffic_density", "traffic"),
        "weather": by("Weather_conditions", "weather"),
        "festival": by("Festival", "festival"),
        "vehicle": by("Type_of_vehicle", "vehicle"),
    }


def busiest_riders(df: pd.DataFrame | None = None, top_n: int = 10) -> pd.DataFrame:
    """The riders carrying the most orders in the dataset, and how they fared.

    Feeds the "is anyone systematically overloaded?" question with real people
    rather than the session's mock riders. A rider near the top of this list
    who ALSO runs a high average delivery time is the signal worth acting on:
    that's someone being given more than they can sustainably carry.
    """
    df = df if df is not None else load_delivery_data()
    out = df.groupby("Delivery_person_ID").agg(
        orders=("Time_taken (min)", "size"),
        avg_delay=("Time_taken (min)", "mean"),
        avg_rating=("Delivery_person_Ratings", "mean"),
        avg_stacked=("multiple_deliveries", "mean"),
    ).reset_index().rename(columns={"Delivery_person_ID": "rider_id"})

    out["city"] = out["rider_id"].str.split("RES").str[0].map(CITY_CODES).fillna("Unknown")
    for col in ("avg_delay", "avg_rating", "avg_stacked"):
        out[col] = out[col].round(2)
    return out.sort_values("orders", ascending=False).head(top_n).reset_index(drop=True)
