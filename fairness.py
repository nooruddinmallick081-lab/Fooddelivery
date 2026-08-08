"""
fairness.py — Rider Fairness Score (Feature #5, the "society & people" one).

Every other feature in this app optimizes for the CUSTOMER's wait time.
This one exists to ask a different question: is the rider being set up to
fail? Stacking too many concurrent orders on a rider with a poorly-
maintained vehicle over a long distance in bad traffic is exactly the
combo that pushes gig workers to rush, take risks, or just burn out — a
real, widely-documented problem in delivery-platform labor conditions.

This is intentionally a simple, transparent weighted score, not another
ML model — a fairness/safety flag should be easy to audit and explain,
not another black box on top of the ETA one. Every number below is one a
rider coordinator could check by hand on the back of a docket, which is
the whole point: if they can't check it, they won't trust it, and a safety
flag nobody trusts doesn't make anybody safer.

The v2 additions here are both about making it *usable* rather than just
correct: the score now shows its working (which factor contributed what),
and there's a fleet-level view, because "is this one rider okay?" is the
wrong question when you're staffing a dinner rush across forty of them.
"""

TRAFFIC_WEIGHT = {"Low": 0, "Medium": 5, "High": 10, "Jam": 15}

# distance past this is where a run starts genuinely eating into a rider's
# shift rather than just being a normal delivery
COMFORTABLE_RADIUS_KM = 10.0

# score thresholds -> (label, advice). Kept as one table so the UI, the tests
# and this docstring can never drift apart on where the boundaries sit.
BANDS = [
    (25, "Fine", "Normal load, nothing to flag."),
    (50, "Watch", "Getting stacked — okay for now, don't add another order."),
    (75, "Overloaded", "Consider reassigning one of this rider's orders."),
    (float("inf"), "High Risk", "Recommend splitting this rider's load immediately."),
]


def band_for(score: float) -> tuple[str, str]:
    """Which band a score falls into. Split out so the fleet view and the
    per-rider view can never disagree about what 'Overloaded' means."""
    for ceiling, label, advice in BANDS:
        if score < ceiling:
            return label, advice
    return BANDS[-1][1], BANDS[-1][2]  # unreachable, inf catches everything


def rider_load_score(multiple_deliveries: float, vehicle_condition: int,
                     distance_km: float, traffic_level: str) -> dict:
    """
    Returns a 0-100 "load score" (higher = rider is more overloaded/at-risk)
    plus a plain-language label, and — new in v2 — the itemised breakdown that
    produced it. Every term is something a coordinator could sanity-check:

      - each extra concurrent delivery stacked on the rider: +25
      - a poorly maintained vehicle (0=worst, 3=best): up to +30
      - distance beyond a comfortable 10km radius: +2 per extra km
      - heavier traffic: up to +15

    The breakdown matters more than it looks. A score of 60 built from "four
    stacked orders" needs a completely different response than a 60 built from
    "one order, terrible bike, long haul in a jam" — the first is a dispatch
    problem, the second is a maintenance one. A single number hides that;
    naming the contributions doesn't.
    """
    if distance_km < 0:
        raise ValueError(f"distance_km can't be negative, got {distance_km}")
    if multiple_deliveries < 0:
        raise ValueError(f"multiple_deliveries can't be negative, got {multiple_deliveries}")

    # clamped so a stray 9 out of the vehicle-condition column can't hand a
    # rider a negative penalty and quietly cancel out a real risk elsewhere
    vehicle_condition = max(0, min(3, int(vehicle_condition)))

    factors = {
        "Concurrent orders": multiple_deliveries * 25,
        "Vehicle condition": (3 - vehicle_condition) * 10,
        "Distance beyond 10 km": max(0.0, distance_km - COMFORTABLE_RADIUS_KM) * 2,
        "Traffic": TRAFFIC_WEIGHT.get(traffic_level, 5),
    }
    raw = sum(factors.values())
    score = min(100.0, round(raw, 1))
    label, advice = band_for(score)

    # what's actually driving this score, biggest first — the UI shows the top
    # one by name so the advice line has a reason attached to it
    drivers = sorted(
        ((name, round(value, 1)) for name, value in factors.items() if value > 0),
        key=lambda kv: -kv[1],
    )

    return {
        "score": score,
        "label": label,
        "advice": advice,
        "factors": {name: round(value, 1) for name, value in factors.items()},
        "drivers": drivers,
        "top_driver": drivers[0][0] if drivers else None,
        "capped": raw > 100,  # the raw number blew past the scale entirely
    }


def fleet_summary(scores: list[dict]) -> dict:
    """Roll a batch of per-rider scores up into one dispatch-desk view.

    A coordinator running a dinner rush doesn't want forty individual numbers,
    they want "three of your riders are in trouble, here's the worst one".
    Takes the dicts that rider_load_score() returns, so there's exactly one
    definition of the score in this codebase and this function can't drift
    away from it.
    """
    if not scores:
        return {
            "n_riders": 0, "mean_score": 0.0, "worst_score": 0.0,
            "n_overloaded": 0, "counts": {label: 0 for _, label, _ in BANDS},
            "headline": "No active riders right now.",
        }

    values = [s["score"] for s in scores]
    counts = {label: 0 for _, label, _ in BANDS}
    for s in scores:
        counts[s["label"]] += 1

    # "overloaded" here means the two bands where we actually recommend an
    # intervention, not merely "worth watching"
    n_overloaded = counts["Overloaded"] + counts["High Risk"]
    mean_score = round(sum(values) / len(values), 1)

    if n_overloaded == 0:
        headline = f"All {len(values)} active rider(s) within a sustainable load."
    elif n_overloaded == 1:
        headline = "1 rider is carrying too much — worth rebalancing before the next order."
    else:
        headline = f"{n_overloaded} riders are carrying too much — rebalance before adding orders."

    return {
        "n_riders": len(values),
        "mean_score": mean_score,
        "worst_score": max(values),
        "n_overloaded": n_overloaded,
        "counts": counts,
        "headline": headline,
    }
