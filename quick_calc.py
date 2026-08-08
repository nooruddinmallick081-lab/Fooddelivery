"""
quick_calc.py — the no-ML "back of napkin" ETA estimate (Feature #1).

Deliberately NOT using the neural net here. This is the honest, transparent
baseline: assumed average road speed for each traffic level, a weather
slowdown multiplier, plus however long the kitchen said the food needs.
Good for comparing against the ML model — "is the neural net actually
smarter than a sensible formula?" is a fair question, and this tab lets a
judge check for themselves.

Keeping this around isn't just a demo prop either. If the model file ever
goes missing or a retrain produces something broken, the app still has a
formula that works and that anyone can verify with a calculator. A product
that degrades to "slightly less clever" beats one that degrades to a
stack trace.
"""

# average achievable speed in city traffic, km/h — rough, defensible numbers,
# not derived from the dataset (that's the whole point, it's the non-ML baseline)
TRAFFIC_SPEED_KMPH = {
    "Low": 32,
    "Medium": 25,
    "High": 18,
    "Jam": 12,
}

# multiplies travel time — bad weather slows a rider down beyond just traffic
WEATHER_SLOWDOWN = {
    "Sunny": 1.00,
    "Cloudy": 1.05,
    "Windy": 1.08,
    "Fog": 1.18,
    "Sandstorms": 1.20,
    "Stormy": 1.30,
    "Unknown": 1.10,
}

# pickup/handoff buffer — parking, finding the door, the customer not picking up
BUFFER_MIN = 3


def quick_eta_estimate(distance_km: float, prep_time_min: float,
                       traffic_level: str, weather: str) -> dict:
    """Distance ÷ assumed speed, slowed for weather, plus prep and a buffer.

    Returns every intermediate number, not just the total, because the entire
    selling point of this function is that you can check its working. If it
    only handed back one number it would be exactly as opaque as the neural
    net, just less accurate.
    """
    if distance_km < 0:
        raise ValueError(f"distance_km can't be negative, got {distance_km}")
    if prep_time_min < 0:
        raise ValueError(f"prep_time_min can't be negative, got {prep_time_min}")

    speed = TRAFFIC_SPEED_KMPH.get(traffic_level, 25)
    slowdown = WEATHER_SLOWDOWN.get(weather, 1.10)

    travel_min = (distance_km / speed) * 60 * slowdown
    total = prep_time_min + travel_min + BUFFER_MIN

    return {
        "travel_min": round(travel_min, 1),
        "prep_min": round(prep_time_min, 1),
        "buffer_min": BUFFER_MIN,
        "total_min": round(total, 1),
        "assumed_speed_kmph": speed,
        "weather_slowdown": slowdown,
        "formula": (
            f"{distance_km:g} km ÷ {speed} km/h × 60 × {slowdown:g} (weather) "
            f"+ {prep_time_min:g} min prep + {BUFFER_MIN} min handoff"
        ),
    }


def compare_with_model(quick_result: dict, model_eta_min: float) -> dict:
    """Put the napkin formula and the neural net side by side, honestly.

    The two disagreeing is expected, not a bug, and the app should say so in
    words rather than leaving someone to wonder which one is broken. The
    biggest single reason they differ: this formula asks you to type in the
    kitchen prep time, while the model learned typical prep time implicitly
    from 45,000 real orders and has no way to separate it back out. So a gap
    of roughly "whatever you typed vs. whatever is typical" is the honest
    expectation, not a defect.
    """
    gap = model_eta_min - quick_result["total_min"]
    if abs(gap) < 3:
        verdict = "Both methods agree closely — the simple formula holds up here."
    elif gap > 0:
        verdict = (
            "The model expects this to run slower than the flat formula does — "
            "it has seen real orders in conditions like these."
        )
    else:
        verdict = (
            "The model expects this to run faster than the flat formula does — "
            "usually the formula's assumed road speed is pessimistic for this route."
        )

    return {
        "quick_min": quick_result["total_min"],
        "model_min": round(model_eta_min, 1),
        "gap_min": round(gap, 1),
        "verdict": verdict,
    }
