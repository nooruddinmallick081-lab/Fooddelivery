"""
marketplace.py — the mock two-sided marketplace layer (customer ordering +
restaurant order-acceptance) that sits ON TOP of the real ETA model and
fairness scorer. The restaurants/menus/test riders below are demo content,
not real business data — but every ETA and fairness number they produce
comes from the same trained model and formula used everywhere else in the
app, nothing here is faked.

Why this exists: a single "type in some numbers, get a prediction" form is
honest but doesn't really show HOW the model would sit inside an actual
delivery app. This simulates both sides of that app (the customer placing
an order, the restaurant accepting it and getting a rider assigned) so the
demo tells a complete story, not just a form.

The v2 rewrite fixed something embarrassing. Rider assignment used to be
`random.choice(RIDERS)` and — worse — the number of orders that rider was
already carrying was ALSO random, `random.choice([0, 1, 2])`. So the Rider
Fairness Score, the one feature this whole project is pitched on, was being
computed from a dice roll rather than from what was actually happening in
the app. It looked fine on screen, which is exactly why it survived so long.
Riders now carry their real session workload, and dispatch actively hands
the next order to the LEAST loaded rider, which is what a platform that
means what it says about fairness would do.
"""

from datetime import datetime

from backend.fairness import rider_load_score
from backend.inference import haversine_km

# Where the demo customer is ordering from. A real coordinate in Bengaluru,
# which is one of the 22 cities that actually appear in the dataset — so the
# distances below are real great-circle distances, not numbers we typed in.
CUSTOMER_LOCATION = (12.9716, 77.5946)

# demo restaurants — name -> cuisine, a real coordinate, a tiny menu, and how
# long the kitchen says it needs. Distances are computed from the coordinates
# with the same haversine function the model's training used.
MOCK_RESTAURANTS = {
    "Punjabi Tadka 🍛": {
        "cuisine": "North Indian", "coords": (12.9352, 77.6245), "prep_min": 18,
        "menu": [("Butter Chicken", 320), ("Dal Makhani", 220), ("Garlic Naan", 60)],
    },
    "Sushi Zen 🍣": {
        "cuisine": "Japanese", "coords": (13.0210, 77.6480), "prep_min": 22,
        "menu": [("California Roll", 380), ("Miso Soup", 150), ("Salmon Nigiri", 420)],
    },
    "Pizza Planet 🍕": {
        "cuisine": "Italian", "coords": (12.9580, 77.6210), "prep_min": 14,
        "menu": [("Margherita", 280), ("Pepperoni", 340), ("Garlic Bread", 120)],
    },
    "Dragon Wok 🥡": {
        "cuisine": "Chinese", "coords": (12.9950, 77.5490), "prep_min": 16,
        "menu": [("Fried Rice", 210), ("Manchurian", 240), ("Spring Rolls", 150)],
    },
}

# fill in each restaurant's distance from the customer once, at import time
for _info in MOCK_RESTAURANTS.values():
    _info["distance_km"] = round(haversine_km(*CUSTOMER_LOCATION, *_info["coords"]), 1)

TEST_CUSTOMERS = ["Aditi Sharma", "Rahul Verma", "Fatima Khan", "Chris Lee"]

TEST_RIDERS = [
    {"name": "Vikram", "rating": 4.7, "age": 26, "vehicle": "motorcycle", "vehicle_condition": 3},
    {"name": "Neha", "rating": 4.3, "age": 23, "vehicle": "scooter", "vehicle_condition": 2},
    {"name": "Suresh", "rating": 4.8, "age": 34, "vehicle": "electric_scooter", "vehicle_condition": 3},
    {"name": "Imran", "rating": 4.1, "age": 21, "vehicle": "bicycle", "vehicle_condition": 2},
]

ORDER_STEPS = ["Placed", "Accepted", "Rider Assigned", "Picked Up", "Delivered"]

# an order is "on the road" between being assigned and being delivered — these
# are the ones that count against a rider's concurrent load
FIRST_ACTIVE_STEP = 1
DELIVERED_STEP = len(ORDER_STEPS) - 1

# the demo runs in one city, so every order is a Metropolitian one
DEMO_AREA_TYPE = "Metropolitian"
DEMO_ORDER_TYPE = "Meal"


def new_order_id(existing: list[dict] | None = None) -> str:
    """Sequential, readable order IDs.

    These used to be `random.randint(1000, 9999)`, which had a real (if small)
    chance of colliding — and since the UI keys its Accept/Reject buttons off
    the order ID, a collision meant two orders sharing a button. Counting is
    both safer and easier to read out loud during a demo.
    """
    return f"FF{1000 + len(existing or []):04d}"


def place_order(customer: str, restaurant: str, cart: list[tuple[str, int, int]],
                traffic_level: str = "Medium", weather: str = "Sunny",
                festival: bool = False, existing_orders: list[dict] | None = None) -> dict:
    """cart is a list of (item_name, qty, unit_price). No ETA/rider yet —
    that only gets assigned once the restaurant accepts, same as reality.

    The conditions the customer picked travel WITH the order. In v1 the
    "simulate conditions" controls existed on the order screen, were dutifully
    collected, and then thrown away — the restaurant screen picked its own
    values and the customer's choice never reached the model at all.
    """
    if not cart:
        raise ValueError("Can't place an empty order.")
    total = sum(qty * price for _, qty, price in cart)
    return {
        "id": new_order_id(existing_orders),
        "customer": customer,
        "restaurant": restaurant,
        "cart": cart,
        "total_price": total,
        "status_idx": 0,  # index into ORDER_STEPS
        "placed_at": datetime.now().strftime("%H:%M:%S"),
        "conditions": {"traffic": traffic_level, "weather": weather, "festival": festival},
        "rider": None,
        "eta_min": None,
        "eta_low_min": None,
        "eta_high_min": None,
        "fairness": None,
    }


def rider_workload(orders: list[dict]) -> dict[str, int]:
    """How many orders each rider is currently carrying, right now, for real.

    "Currently" means assigned but not yet delivered. This is the number the
    fairness score needs, and the number v1 was faking with a dice roll.
    """
    load = {rider["name"]: 0 for rider in TEST_RIDERS}
    for order in orders:
        rider = order.get("rider")
        if rider and FIRST_ACTIVE_STEP <= order["status_idx"] < DELIVERED_STEP:
            load[rider["name"]] = load.get(rider["name"], 0) + 1
    return load


def pick_rider(orders: list[dict]) -> tuple[dict, int]:
    """Hand the next order to whoever is carrying the least.

    Returns (rider, orders_they_already_have). Ties break on rating, so a
    quiet shift doesn't always land on whoever happens to be first in the
    list. This is the whole thesis of the project expressed in four lines:
    a dispatcher that balances load is the intervention, the fairness score
    is just how you notice you need one.
    """
    load = rider_workload(orders)
    rider = min(TEST_RIDERS, key=lambda r: (load[r["name"]], -r["rating"]))
    return rider, load[rider["name"]]


def accept_order(order: dict, model, traffic_level: str | None = None,
                 weather: str | None = None, festival: bool | None = None,
                 all_orders: list[dict] | None = None) -> dict:
    """Restaurant accepts -> the least-loaded rider is assigned -> we run the
    REAL model + fairness scorer for that rider/restaurant/order combo.

    Conditions default to whatever the customer picked when they ordered; the
    restaurant can override them (it's the restaurant that knows what the
    weather is doing outside right now, not the customer's dropdown from ten
    minutes ago), but nothing is silently discarded either way.
    """
    conditions = order.get("conditions", {})
    traffic_level = traffic_level or conditions.get("traffic", "Medium")
    weather = weather or conditions.get("weather", "Sunny")
    if festival is None:
        festival = conditions.get("festival", False)

    restaurant = MOCK_RESTAURANTS[order["restaurant"]]
    all_orders = all_orders if all_orders is not None else []
    rider, current_load = pick_rider(all_orders)

    prediction = model.predict_with_interval(
        restaurant["distance_km"], traffic_level, current_load,
        rider["vehicle_condition"], rider["rating"], rider["age"], festival,
        DEMO_AREA_TYPE, weather, DEMO_ORDER_TYPE, rider["vehicle"],
    )
    fairness = rider_load_score(
        current_load, rider["vehicle_condition"], restaurant["distance_km"], traffic_level
    )

    order["rider"] = rider
    order["rider_load_at_assignment"] = current_load
    order["eta_min"] = round(prediction["eta_min"], 1)
    order["eta_low_min"] = round(prediction["low_min"], 1)
    order["eta_high_min"] = round(prediction["high_min"], 1)
    order["eta_warnings"] = prediction["warnings"]
    order["conditions"] = {"traffic": traffic_level, "weather": weather, "festival": festival}
    order["fairness"] = fairness
    order["status_idx"] = FIRST_ACTIVE_STEP
    return order


def advance_order(order: dict) -> dict:
    """Move an order one step along the tracker, stopping at Delivered."""
    order["status_idx"] = min(order["status_idx"] + 1, DELIVERED_STEP)
    return order


def kitchen_summary(orders: list[dict], restaurant: str) -> dict:
    """One restaurant's whole board at a glance, plus the fleet fairness view.

    The restaurant screen was working all of this out inline with four list
    comprehensions over the same list; pulling it into the backend means the
    numbers on that screen are testable without booting Streamlit, which is
    the rule the rest of this project already follows.
    """
    mine = [o for o in orders if o["restaurant"] == restaurant]
    incoming = [o for o in mine if o["status_idx"] == 0]
    active = [o for o in mine if FIRST_ACTIVE_STEP <= o["status_idx"] < DELIVERED_STEP]
    delivered = [o for o in mine if o["status_idx"] == DELIVERED_STEP]

    revenue = sum(o["total_price"] for o in delivered)
    etas = [o["eta_min"] for o in active if o["eta_min"] is not None]

    return {
        "incoming": incoming,
        "active": active,
        "delivered": delivered,
        "n_incoming": len(incoming),
        "n_active": len(active),
        "n_delivered": len(delivered),
        "revenue_delivered": revenue,
        "avg_active_eta": round(sum(etas) / len(etas), 1) if etas else None,
        # fleet fairness spans every restaurant on purpose — riders are shared,
        # so one kitchen's dispatch decisions land on another kitchen's rider
        "fleet": _fleet_view(orders),
    }


def _fleet_view(orders: list[dict]) -> dict:
    """Current fairness picture across every rider on the road right now."""
    from backend.fairness import fleet_summary

    load = rider_workload(orders)
    per_rider = []
    for rider in TEST_RIDERS:
        active = [
            o for o in orders
            if o.get("rider") and o["rider"]["name"] == rider["name"]
            and FIRST_ACTIVE_STEP <= o["status_idx"] < DELIVERED_STEP
        ]
        if not active:
            continue
        # score the rider against their longest current run, which is the one
        # setting how stretched they actually are
        longest = max(MOCK_RESTAURANTS[o["restaurant"]]["distance_km"] for o in active)
        worst_traffic = max(
            (o["conditions"]["traffic"] for o in active),
            key=lambda t: {"Low": 0, "Medium": 1, "High": 2, "Jam": 3}.get(t, 1),
        )
        score = rider_load_score(
            load[rider["name"]] - 1, rider["vehicle_condition"], longest, worst_traffic
        )
        per_rider.append({"rider": rider["name"], "orders": load[rider["name"]], **score})

    return {"per_rider": per_rider, **fleet_summary(per_rider)}
