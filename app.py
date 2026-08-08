"""
app.py — FeedForward's one and only Streamlit page.

Structure, v4: a sidebar hamburger-style nav (Streamlit collapses this
behind a "»" toggle automatically on narrow/mobile screens — that's our
hamburger menu, native behavior, nothing hand-rolled) instead of top tabs.
Landing page is the live delay map. Two mock-marketplace screens (Order
Food / Restaurant Hub) simulate both sides of a real delivery app, both
driven by the SAME real model and fairness scorer as everywhere else — see
backend/marketplace.py for the glue, backend/inference.py +
backend/fairness.py for the actual math, nothing here is faked.

The rule this file follows, same as BeejBank: it NEVER does the maths
itself. Every number on screen came out of a backend function that can be
tested without opening a browser. If you find arithmetic in here that
isn't laying out a chart, it's in the wrong file.

Run it with: streamlit run frontend/app.py   (or double-click run.bat on
the portable Windows build)
"""

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.data_utils import (
    city_performance, condition_impact, dataset_summary,
    delay_hotspots, load_delivery_data, rush_hour_profile,
)
from backend.fairness import rider_load_score
from backend.inference import ETAModel, ModelSchemaError
from backend.marketplace import (
    MOCK_RESTAURANTS, ORDER_STEPS, TEST_CUSTOMERS,
    accept_order, advance_order, kitchen_summary, place_order,
)
from backend.quick_calc import compare_with_model, quick_eta_estimate

BRAND_PRIMARY = "#E23744"
BRAND_INK = "#2B1810"

st.set_page_config(page_title="FeedForward — Smart Delivery ETA", page_icon="🍔", layout="wide")

_css_path = Path(__file__).resolve().parent / "assets" / "style.css"
st.markdown(f"<style>{_css_path.read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# small render helpers
# ---------------------------------------------------------------------------
def html(markup: str):
    """Render a raw HTML block, flattened to a single line first.

    This exists because of a bug that headless tests can't see and a browser
    shows instantly. Streamlit runs our markup through a Markdown parser even
    with unsafe_allow_html=True, and Markdown has two rules that bite here:
    a blank line ends an HTML block, and a line indented by four or more
    spaces is a CODE block. So a nicely-indented f-string whose conditional
    piece evaluates to "" leaves a whitespace-only line, and every line after
    it gets rendered as literal source code on the page.

    That's exactly what happened to the top bar: when nobody had an order on
    the road, the "on the road" chip collapsed to "", and the app printed
    `<div class="ff-live">LIVE MODEL</div></div></div>` in a grey code box in
    the corner of every single screen.

    Stripping every line and joining removes both hazards at once, so callers
    can keep writing readable indented HTML.
    """
    flat = "".join(line.strip() for line in markup.splitlines())
    st.markdown(flat, unsafe_allow_html=True)


def card_row(cards: list[dict]):
    parts = ['<div class="ff-card-grid">']
    for c in cards:
        tone_class = f" ff-{c['tone']}" if c.get("tone") else ""
        sub_html = f'<div class="ff-sub">{c["sub"]}</div>' if c.get("sub") else ""
        parts.append(
            f'<div class="ff-card{tone_class}">'
            f'<div class="ff-icon">{c.get("icon", "")}</div>'
            f'<div class="ff-value">{c["value"]}</div>'
            f'<div class="ff-label">{c["label"]}</div>'
            f'{sub_html}</div>'
        )
    parts.append("</div>")
    html("".join(parts))


def glow_metric(emoji: str, value: str, label: str, band: str | None = None):
    """The one big headline stat per screen — a static card with a colored
    border whose SHADOW gently pulses. An earlier version rotated a conic
    gradient behind it; even though the text never moved, a constantly
    cycling border reads as "the whole thing is spinning", so it's gone.

    `band` is the confidence range printed under the number. Showing a
    single figure with no range implies a precision this model doesn't have.
    """
    band_html = f'<div class="ff-glow-band">{band}</div>' if band else ""
    html(f"""<div class="ff-glow-wrap">
               <div class="ff-glow-emoji">{emoji}</div>
               <div>
                 <div class="ff-glow-value">{value}</div>
                 <div class="ff-glow-label">{label}</div>
                 {band_html}
               </div>
             </div>""")


def skeleton(height: int = 220):
    html(f'<div class="ff-skeleton" style="height:{height}px"></div>')


def section(title: str, sub: str | None = None):
    html(f'<div class="ff-section-title">{title}</div>')
    if sub:
        html(f'<div class="ff-section-sub">{sub}</div>')


def stepper(current_idx: int, steps: list[str] = ORDER_STEPS):
    parts = ['<div class="ff-stepper">']
    for i, step in enumerate(steps):
        cls = "ff-done" if i < current_idx else ("ff-current" if i == current_idx else "")
        mark = "✓" if i < current_idx else str(i + 1)
        parts.append(
            f'<div class="ff-step {cls}"><div class="ff-step-dot">{mark}</div>'
            f'<div class="ff-step-label">{step}</div></div>'
        )
    parts.append("</div>")
    html("".join(parts))


def fairness_tone(score: float) -> str:
    return "danger" if score >= 75 else ("warn" if score >= 50 else "good")


def style_chart(fig, height: int = 380):
    """One place that decides what our charts look like, so eight different
    charts can't drift into eight different fonts and grid colours."""
    fig.update_layout(
        height=height,
        margin=dict(l=10, r=10, t=20, b=10),
        plot_bgcolor="white",
        paper_bgcolor="white",
        font=dict(family="-apple-system, Segoe UI, Roboto, sans-serif",
                  size=13, color=BRAND_INK),
        hoverlabel=dict(bgcolor="white", font_size=13),
    )
    fig.update_xaxes(gridcolor="#F3E4DA", zerolinecolor="#F3E4DA")
    fig.update_yaxes(gridcolor="#F3E4DA", zerolinecolor="#F3E4DA")
    return fig


def goto(page_label: str):
    """Queue a page switch for the NEXT run.

    We can't just write to `ff_nav_radio` here: Streamlit refuses to let you
    modify a widget's own key after that widget has been created, and by the
    time a button on the page has been clicked, the sidebar radio already
    exists. Doing it directly is what made both landing-page buttons throw a
    StreamlitAPIException instead of navigating anywhere. So we park the
    request under a different key and apply it at the top of the next run,
    before the radio is built.
    """
    st.session_state["ff_pending_nav"] = page_label
    st.rerun()


@st.cache_resource
def get_model():
    return ETAModel()


@st.cache_data(show_spinner=False)
def get_data():
    return load_delivery_data()


@st.cache_data(show_spinner=False)
def get_hotspots():
    """Cached because the landing page rebuilds this on every single rerun
    otherwise — it's a groupby over 42,000 rows, and nothing about it changes
    between clicks."""
    return delay_hotspots()


@st.cache_data(show_spinner=False)
def get_city_performance():
    return city_performance()


@st.cache_data(show_spinner=False)
def get_rush_profile():
    return rush_hour_profile()


@st.cache_data(show_spinner=False)
def get_condition_impact():
    return condition_impact()


@st.cache_data(show_spinner=False)
def get_summary():
    return dataset_summary()


# If the model files are missing or out of sync with each other, say so in
# plain language and stop. A blank page with a stack trace helps nobody, and
# an app that predicts confident garbage from mismatched weights helps less.
try:
    model = get_model()
except (FileNotFoundError, ModelSchemaError) as exc:
    st.error("FeedForward can't start — the trained model files are missing or out of sync.")
    st.code(str(exc))
    st.info("Fix: restore `model/weights.npz` + `model/meta.json` from the repo, "
            "or re-run `model/train_model.ipynb` to regenerate both together.")
    st.stop()

df = get_data()
summary = get_summary()
CARD = model.model_card()

if "ff_orders" not in st.session_state:
    st.session_state.ff_orders = []

WEATHER_OPTS = model.weather_categories
CITY_OPTS = model.city_categories
ORDER_OPTS = model.order_categories
VEHICLE_OPTS = model.vehicle_categories
TRAFFIC_OPTS = model.traffic_levels

# Slider limits come from the model's own record of what it was trained on,
# never typed in by hand. That way a retrain can't leave the UI offering
# values the model has no business answering — the same fix BeejBank made
# after finding its fertilizer slider ran 24 standard deviations past the data.
DIST_MIN = float(np.ceil(CARD["input_domain"]["distance_km"]["min"] * 10) / 10)
DIST_MAX = float(np.floor(CARD["input_domain"]["distance_km"]["max"] * 10) / 10)
DIST_DEFAULT = round((DIST_MIN + DIST_MAX) / 2, 1)


def default_index(options: list, wanted: str) -> int:
    return options.index(wanted) if wanted in options else 0


# ---------------------------------------------------------------------------
# splash / login gate — cosmetic only, no real auth. Makes opening the app
# feel like opening an actual delivery app instead of a bare dashboard.
# ---------------------------------------------------------------------------
if "ff_started" not in st.session_state:
    st.session_state.ff_started = False

if not st.session_state.ff_started:
    with st.container(key="ff_splash"):
        html('<div class="ff-splash-emoji">🍔➡️</div>')
        html('<div class="ff-splash-title">FeedForward</div>')
        html('<div class="ff-splash-tagline">Smart delivery ETAs that respect the rider too.</div>')
        # NOTE: this card has to be a real st.container with a key, not a raw
        # <div> opened in one st.markdown and closed in another. Streamlit
        # renders every markdown block as its own isolated node and auto-closes
        # any tag you leave hanging, so the old version produced an empty card
        # floating next to unstyled widgets. Same lesson as the buttons below:
        # you cannot wrap a real widget in injected HTML.
        with st.container(key="ff_splash_card"):
            name = st.text_input("Your name", placeholder="What should we call you?",
                                 label_visibility="collapsed", key="ff_name_input")
            started = st.button("Get Started 🚀", key="ff_start_btn", type="primary")
        html(f"""<div class="ff-splash-badges">
                   <span>🛵 {summary['n_orders']:,} real orders</span>
                   <span>📶 works offline</span>
                   <span>⚖️ rider-fair by design</span>
                 </div>""")
        if started:
            st.session_state.ff_started = True
            st.session_state.ff_user = name.strip() or "there"
            st.rerun()
    st.stop()

# ---------------------------------------------------------------------------
# sidebar — the hamburger menu (Streamlit auto-collapses this on mobile)
# ---------------------------------------------------------------------------
NAV = [
    "🗺️ Live Map",
    "🛒 Order Food",
    "🏪 Restaurant Hub",
    "📦 ETA Predictor",
    "🍽️ Quick Calculator",
    "🎛️ What-If Simulator",
    "📊 Rush Hours",
    "🔍 Why This ETA?",
]

# apply any queued navigation BEFORE the radio is instantiated — see goto()
if "ff_pending_nav" in st.session_state:
    st.session_state["ff_nav_radio"] = st.session_state.pop("ff_pending_nav")

with st.sidebar:
    html('<div class="ff-side-logo">🍔➡️ <span>FeedForward</span></div>')
    html(f'<div class="ff-side-user">Signed in as <b>{st.session_state.get("ff_user", "there")}</b></div>')
    page = st.radio("Navigate", NAV, label_visibility="collapsed", key="ff_nav_radio")
    html(f"""<div class="ff-side-foot">
               <div class="ff-side-stat"><b>{CARD['mae_minutes']:.1f} min</b><span>typical error</span></div>
               <div class="ff-side-stat"><b>{summary['n_orders']:,}</b><span>orders trained on</span></div>
             </div>""")

n_live = sum(1 for o in st.session_state.ff_orders if 0 < o["status_idx"] < len(ORDER_STEPS) - 1)
html(f"""<div class="ff-topbar">
           <div class="ff-greet">Hey <b>{st.session_state.get("ff_user", "there")}</b> 👋</div>
           <div class="ff-topbar-right">
             {'<div class="ff-chip">🛵 ' + str(n_live) + ' on the road</div>' if n_live else ''}
             <div class="ff-live">LIVE MODEL</div>
           </div>
         </div>""")

# ---------------------------------------------------------------------------
# PAGE: Live Map — the landing page
# ---------------------------------------------------------------------------
if page == "🗺️ Live Map":
    html("""<div class="ff-hero">
               <h1>🍔➡️ <span class="ff-shine-text">FeedForward</span></h1>
               <p>Delivery-time prediction that also looks out for the rider — not just
               the customer's countdown timer.</p>
             </div>""")

    card_row([
        {"icon": "🛵", "label": "Real orders analysed", "value": f"{summary['n_orders']:,}",
         "sub": f"{summary['n_riders']:,} riders · {summary['n_cities']} cities"},
        {"icon": "🎯", "label": "Typical error", "value": f"±{CARD['mae_minutes']:.1f} min",
         "sub": f"{CARD['pct_within_10_min']:.0f}% within 10 min", "tone": "accent"},
        {"icon": "⏱️", "label": "Average delivery", "value": f"{summary['mean_delivery_min']} min",
         "sub": f"range {summary['fastest_min']}–{summary['slowest_min']} min"},
    ])

    cta1, cta2 = st.columns(2)
    with cta1:
        if st.button("🛒 I'm hungry — order food", key="cta_order", type="primary"):
            goto("🛒 Order Food")
    with cta2:
        if st.button("🏪 I run a kitchen — check my queue", key="cta_rest", type="primary"):
            goto("🏪 Restaurant Hub")

    section(
        "Where are deliveries slowest?",
        "Real restaurant GPS locations, grouped into zones and named by the city they "
        "sit in — flags areas that may be underserved, not just 'traffic is bad' in "
        "the abstract.",
    )

    map_slot = st.empty()
    with map_slot.container():
        skeleton(420)
    hotspots = get_hotspots()
    map_slot.empty()

    if hotspots.empty:
        st.warning("Not enough usable location data to build the map.")
    else:
        fig = px.scatter_geo(
            hotspots, lat="grid_lat", lon="grid_lon",
            color="avg_delay", size="order_count", hover_name="city",
            color_continuous_scale=["#FFD9A0", "#FF6B2C", "#B3261E"],
            hover_data={"avg_delay": ":.1f", "order_count": ":,",
                        "grid_lat": False, "grid_lon": False},
            labels={"avg_delay": "Avg delivery (min)", "order_count": "Orders"},
            size_max=34,
        )
        # fit the frame to India instead of drawing the whole globe with a
        # cluster of dots in one corner of it
        fig.update_geos(
            scope="asia", resolution=50,
            lataxis_range=[6, 36], lonaxis_range=[67, 92],
            showcountries=True, countrycolor="#E4CFC2",
            showland=True, landcolor="#FFF6F0",
            showocean=True, oceancolor="#F2F7FA",
            showlakes=False, showframe=False, coastlinecolor="#E4CFC2",
        )
        fig.update_layout(height=470, margin=dict(l=0, r=0, t=6, b=0),
                          coloraxis_colorbar=dict(title="min", thickness=12))
        st.plotly_chart(fig, width="stretch")
        st.caption(
            f"Zones grid-bucketed at ~0.5° resolution, only cells with 15+ orders shown. "
            f"{summary['n_null_island']:,} orders whose coordinates were recorded as (0, 0) are "
            f"excluded — they aren't a place, they're a missing value, and they used to plot "
            f"as a giant phantom hotspot in the Atlantic."
        )

    section("Slowest cities", "Same data, ranked — the map shows where, this shows who.")
    cities = get_city_performance()
    top_cities = cities.head(8)
    fig = px.bar(
        top_cities.sort_values("avg_delay"), x="avg_delay", y="city", orientation="h",
        color="avg_delay", color_continuous_scale=["#FFD9A0", "#B3261E"],
        labels={"avg_delay": "Average delivery time (min)", "city": ""},
        hover_data={"order_count": ":,", "jam_share_pct": ":.1f"},
    )
    fig.update_layout(coloraxis_showscale=False)
    st.plotly_chart(style_chart(fig, 330), width="stretch")
    st.caption(
        "City comes from the rider ID prefix (`JAPRES09DEL03` → JAP → Jaipur). We "
        "confirmed all 22 codes by checking each one's median coordinate landed on the "
        "city we expected."
    )

# ---------------------------------------------------------------------------
# PAGE: Order Food — customer mock UI
# ---------------------------------------------------------------------------
elif page == "🛒 Order Food":
    section(
        "Order food 🛵",
        "Pick a test account, pick a restaurant, place a mock order — the ETA and rider "
        "you get back are real model output, not scripted.",
    )

    customer = st.selectbox("Ordering as", TEST_CUSTOMERS, key="order_customer")

    st.markdown("**Restaurants near you**")
    names = list(MOCK_RESTAURANTS)
    # two rows of two on purpose — four across is unreadable on a phone, and
    # Streamlit columns don't wrap on their own
    for row_start in range(0, len(names), 2):
        cols = st.columns(2)
        for col, rname in zip(cols, names[row_start:row_start + 2]):
            rinfo = MOCK_RESTAURANTS[rname]
            with col:
                selected_now = st.session_state.get("order_restaurant") == rname
                html(f"""<div class="ff-restaurant-card{' ff-r-selected' if selected_now else ''}">
                          <div class="ff-r-emoji">{rname.split()[-1]}</div>
                          <div class="ff-r-name">{rname.rsplit(" ", 1)[0]}</div>
                          <div class="ff-r-meta">{rinfo['cuisine']} · {rinfo['distance_km']} km ·
                            {rinfo['prep_min']} min prep</div>
                        </div>""")
                if st.button("Selected ✓" if selected_now else "Select",
                             key=f"sel_{rname}",
                             type="primary" if selected_now else "secondary"):
                    st.session_state["order_restaurant"] = rname
                    st.rerun()

    selected = st.session_state.get("order_restaurant")
    if selected:
        st.divider()
        st.markdown(f"**Menu — {selected}**")
        cart = []
        for item, price in MOCK_RESTAURANTS[selected]["menu"]:
            c1, c2 = st.columns([3, 1])
            with c1:
                html(f'<div class="ff-menu-item"><span class="ff-mi-name">{item}</span>'
                     f'<span class="ff-mi-price">₹{price}</span></div>')
            with c2:
                qty = st.number_input("Qty", min_value=0, max_value=5, value=0,
                                      key=f"qty_{selected}_{item}", label_visibility="collapsed")
            if qty > 0:
                cart.append((item, qty, price))

        with st.expander("⚙️ Simulate conditions (traffic, weather, festival)"):
            st.caption(
                "These travel with the order to the restaurant and into the model. "
                "Festival days run about 19 minutes longer in the real data — worth trying."
            )
            oc1, oc2, oc3 = st.columns(3)
            with oc1:
                o_traffic = st.selectbox("Traffic", TRAFFIC_OPTS,
                                         index=default_index(TRAFFIC_OPTS, "Medium"), key="o_traffic")
            with oc2:
                o_weather = st.selectbox("Weather", WEATHER_OPTS,
                                         index=default_index(WEATHER_OPTS, "Sunny"), key="o_weather")
            with oc3:
                o_festival = st.toggle("Festival day", key="o_festival")

        total = sum(q * p for _, q, p in cart)
        st.markdown(f"### Cart total: ₹{total}")

        if st.button("Place Order 🛒", key="place_order_btn", disabled=(total == 0), type="primary"):
            with st.spinner("Sending your order to the restaurant..."):
                time.sleep(0.4)
                order = place_order(customer, selected, cart, o_traffic, o_weather,
                                    o_festival, st.session_state.ff_orders)
            st.session_state.ff_orders.append(order)
            st.success(f"Order {order['id']} placed! Track it below.")
            st.info("The restaurant has to accept it before a rider is assigned — "
                    "head to **🏪 Restaurant Hub** to play the other side.")

    st.divider()
    section("My orders")
    my_orders = [o for o in st.session_state.ff_orders if o["customer"] == customer]
    if not my_orders:
        st.info("No orders yet — place one above.")
    for o in reversed(my_orders):
        cond = o["conditions"]
        html(f"""<div class="ff-order-card">
                  <div class="ff-oc-top"><span>{o['restaurant']}</span><span>₹{o['total_price']}</span></div>
                  <div class="ff-oc-sub">Order {o['id']} · placed {o['placed_at']} ·
                    {cond['traffic']} traffic · {cond['weather']}
                    {'· 🎉 festival' if cond['festival'] else ''}</div>
                </div>""")
        stepper(o["status_idx"])
        if o["rider"]:
            card_row([
                {"icon": "🛵", "label": "Rider", "value": o["rider"]["name"],
                 "sub": f"⭐ {o['rider']['rating']} · {o['rider']['vehicle'].replace('_', ' ')}"},
                {"icon": "⏱️", "label": "ETA", "value": f"{o['eta_min']} min",
                 "sub": f"usually {o['eta_low_min']}–{o['eta_high_min']} min", "tone": "accent"},
            ])

# ---------------------------------------------------------------------------
# PAGE: Restaurant Hub — partner mock UI
# ---------------------------------------------------------------------------
elif page == "🏪 Restaurant Hub":
    section(
        "Restaurant hub 🏪",
        "Accept incoming orders, get a rider assigned automatically, and see their "
        "fairness-load score before you commit them to the run.",
    )

    restaurant = st.selectbox("Managing", list(MOCK_RESTAURANTS), key="hub_restaurant")
    board = kitchen_summary(st.session_state.ff_orders, restaurant)

    card_row([
        {"icon": "📥", "label": "Incoming", "value": board["n_incoming"]},
        {"icon": "🔥", "label": "On the road", "value": board["n_active"],
         "sub": f"avg ETA {board['avg_active_eta']} min" if board["avg_active_eta"] else None,
         "tone": "accent"},
        {"icon": "✅", "label": "Delivered", "value": board["n_delivered"],
         "sub": f"₹{board['revenue_delivered']:,} earned"},
    ])

    st.markdown(f"**📥 Incoming orders ({board['n_incoming']})**")
    if not board["incoming"]:
        st.info("No new orders right now. Go place one from Order Food, or switch restaurants.")
    for o in board["incoming"]:
        cond = o["conditions"]
        html(f"""<div class="ff-order-card">
                  <div class="ff-oc-top"><span>{o['customer']}</span><span>₹{o['total_price']}</span></div>
                  <div class="ff-oc-sub">Order {o['id']} · {len(o['cart'])} item(s) ·
                    customer reported {cond['traffic']} traffic, {cond['weather']}</div>
                </div>""")
        ac1, ac2, ac3 = st.columns([1, 1, 1])
        with ac1:
            a_traffic = st.selectbox("Traffic now", TRAFFIC_OPTS,
                                     index=default_index(TRAFFIC_OPTS, cond["traffic"]),
                                     key=f"traffic_{o['id']}")
        with ac2:
            a_weather = st.selectbox("Weather now", WEATHER_OPTS,
                                     index=default_index(WEATHER_OPTS, cond["weather"]),
                                     key=f"weather_{o['id']}")
        with ac3:
            a_festival = st.toggle("Festival day", value=cond["festival"], key=f"fest_{o['id']}")

        b1, b2 = st.columns([2, 1])
        with b1:
            if st.button("✅ Accept & assign rider", key=f"accept_{o['id']}", type="primary"):
                with st.spinner("Finding the least-loaded rider..."):
                    time.sleep(0.4)
                    accept_order(o, model, a_traffic, a_weather, a_festival,
                                 st.session_state.ff_orders)
                st.rerun()
        with b2:
            if st.button("❌ Reject", key=f"reject_{o['id']}", type="secondary"):
                st.session_state.ff_orders.remove(o)
                st.rerun()
        st.divider()

    st.markdown(f"**🔥 Active orders ({board['n_active']})**")
    for o in board["active"]:
        html(f"""<div class="ff-order-card">
                  <div class="ff-oc-top"><span>{o['customer']}</span><span>₹{o['total_price']}</span></div>
                  <div class="ff-oc-sub">Order {o['id']} · {o['conditions']['traffic']} traffic ·
                    {o['conditions']['weather']}</div>
                </div>""")
        stepper(o["status_idx"])
        fair = o["fairness"]
        card_row([
            {"icon": "🛵", "label": "Rider", "value": o["rider"]["name"],
             "sub": f"⭐ {o['rider']['rating']} · carrying {o['rider_load_at_assignment'] + 1}"},
            {"icon": "⏱️", "label": "ETA", "value": f"{o['eta_min']} min",
             "sub": f"usually {o['eta_low_min']}–{o['eta_high_min']} min", "tone": "accent"},
            {"icon": "⚖️", "label": "Rider load", "value": f"{fair['score']}/100",
             "sub": fair["label"], "tone": fairness_tone(fair["score"])},
        ])
        if fair["score"] >= 50:
            st.warning(f"⚖️ {fair['advice']} Biggest factor: **{fair['top_driver']}**.", icon="⚠️")
        for w in o.get("eta_warnings", []):
            st.caption(f"⚠️ {w}")
        if st.button("▶ Advance status", key=f"advance_{o['id']}", type="primary"):
            advance_order(o)
            st.rerun()
        st.divider()

    fleet = board["fleet"]
    section("⚖️ Fleet fairness", "Riders are shared between kitchens, so this spans all of them.")
    if not fleet["per_rider"]:
        st.info("No riders on the road yet — accept an order to dispatch one.")
    else:
        tone = "danger" if fleet["n_overloaded"] else "good"
        card_row([
            {"icon": "🛵", "label": "Riders active", "value": fleet["n_riders"]},
            {"icon": "📊", "label": "Mean load score", "value": f"{fleet['mean_score']}/100"},
            {"icon": "🚨", "label": "Need rebalancing", "value": fleet["n_overloaded"], "tone": tone},
        ])
        st.markdown(f"**{fleet['headline']}**")
        rows = []
        for r in sorted(fleet["per_rider"], key=lambda r: -r["score"]):
            rows.append(
                f'<div class="ff-lb-row"><span class="ff-lb-name">{r["rider"]}</span>'
                f'<span class="ff-lb-mid">{r["orders"]} order(s)</span>'
                f'<span class="ff-badge ff-{fairness_tone(r["score"])}">{r["score"]}/100 · {r["label"]}</span></div>'
            )
        html("".join(rows))
        st.caption(
            "Dispatch always hands the next order to the least-loaded rider — that's the "
            "intervention. The score is just how you notice you needed one."
        )

# ---------------------------------------------------------------------------
# PAGE: ETA Predictor (Feature #2) + Rider Fairness (Feature #5, bundled)
# ---------------------------------------------------------------------------
elif page == "📦 ETA Predictor":
    section(
        "AI-predicted delivery time",
        f"Trained on {summary['n_orders']:,} real Zomato delivery records — knows more "
        f"context than a flat formula.",
    )

    f1, f2, f3 = st.columns(3)
    with f1:
        f_dist = st.slider("Distance (km)", DIST_MIN, DIST_MAX, DIST_DEFAULT, step=0.1, key="f_dist")
        f_traffic = st.selectbox("Traffic", TRAFFIC_OPTS,
                                 index=default_index(TRAFFIC_OPTS, "Medium"), key="f_traffic")
    with f2:
        f_weather = st.selectbox("Weather", WEATHER_OPTS,
                                 index=default_index(WEATHER_OPTS, "Sunny"), key="f_weather")
        f_city = st.selectbox("Area type", CITY_OPTS, key="f_city")
    with f3:
        f_order = st.selectbox("Order type", ORDER_OPTS,
                               index=default_index(ORDER_OPTS, "Meal"), key="f_order")
        f_vehicle = st.selectbox("Rider's vehicle", VEHICLE_OPTS,
                                 index=default_index(VEHICLE_OPTS, "motorcycle"), key="f_vehicle")

    f4, f5, f6, f7 = st.columns(4)
    with f4:
        f_multi = st.slider("Rider's other concurrent orders", 0, 3, 1, key="f_multi")
    with f5:
        f_vcond = st.slider("Vehicle condition (0=poor, 3=great)", 0, 3, 1, key="f_vcond")
    with f6:
        f_rating = st.slider("Rider rating", 1.0, 5.0, 4.6, step=0.1, key="f_rating")
    with f7:
        f_age = st.slider("Rider age", 18, 50, 30, key="f_age")

    f_festival = st.toggle("Festival day", value=False, key="f_festival")

    if st.button("🔮 Predict ETA", type="primary"):
        with st.spinner("Crunching traffic, weather & rider data..."):
            time.sleep(0.4)
            pred = model.predict_with_interval(
                f_dist, f_traffic, f_multi, f_vcond, f_rating, f_age,
                f_festival, f_city, f_weather, f_order, f_vehicle,
            )
            load = rider_load_score(f_multi, f_vcond, f_dist, f_traffic)
            baseline = quick_eta_estimate(
                f_dist, MOCK_RESTAURANTS["Pizza Planet 🍕"]["prep_min"], f_traffic, f_weather
            )
        st.session_state["last_prediction"] = dict(
            distance_km=f_dist, traffic_level=f_traffic, multiple_deliveries=f_multi,
            vehicle_condition=f_vcond, rider_rating=f_rating, rider_age=f_age,
            festival=f_festival, city=f_city, weather=f_weather,
            order_type=f_order, vehicle_type=f_vehicle, pred=pred["eta_min"],
        )

    last = st.session_state.get("last_prediction")
    if last:
        pred = model.predict_with_interval(
            last["distance_km"], last["traffic_level"], last["multiple_deliveries"],
            last["vehicle_condition"], last["rider_rating"], last["rider_age"],
            last["festival"], last["city"], last["weather"],
            last["order_type"], last["vehicle_type"],
        )
        load = rider_load_score(last["multiple_deliveries"], last["vehicle_condition"],
                                last["distance_km"], last["traffic_level"])

        glow_metric(
            "⏱️", f"{pred['eta_min']:,.1f} min", "Predicted delivery time",
            band=f"80% of deliveries like this land between "
                 f"<b>{pred['low_min']:.0f}</b> and <b>{pred['high_min']:.0f}</b> min",
        )
        for w in pred["warnings"]:
            st.warning(w, icon="📏")

        traffic_acc = model.accuracy_for_traffic(last["traffic_level"])
        card_row([
            {"icon": "⚖️", "label": "Rider load score", "value": f"{load['score']}/100",
             "sub": load["label"], "tone": fairness_tone(load["score"])},
            {"icon": "🎯", "label": f"Accuracy in {last['traffic_level']} traffic",
             "value": f"±{traffic_acc['mae_minutes']:.1f} min" if traffic_acc else "—",
             "sub": f"measured on {traffic_acc['n']:,} held-out orders" if traffic_acc else None},
        ])

        if load["score"] >= 50:
            st.warning(f"⚖️ **{load['label']}** — {load['advice']}", icon="⚠️")
        else:
            st.success(f"⚖️ **{load['label']}** — {load['advice']}")

        with st.expander("⚖️ How that load score was built"):
            for factor_name, value in load["factors"].items():
                st.markdown(f"- **{factor_name}**: +{value:g}")
            st.caption("Plain weighted arithmetic, on purpose — a safety flag a coordinator "
                       "can't check by hand is a safety flag nobody trusts.")

        st.info("Head over to **🔍 Why This ETA?** to see what actually drove this prediction.")

# ---------------------------------------------------------------------------
# PAGE: Quick Calculator (no-ML baseline, Feature #1)
# ---------------------------------------------------------------------------
elif page == "🍽️ Quick Calculator":
    section(
        "Back-of-napkin ETA",
        "No AI, just a sensible formula — assumed road speed per traffic level + a "
        "weather slowdown + however long the kitchen says the food needs. Fully transparent.",
    )

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        q_dist = st.slider("Distance (km)", DIST_MIN, DIST_MAX, DIST_DEFAULT, step=0.1, key="q_dist")
    with c2:
        q_prep = st.number_input("Kitchen prep time (min)", min_value=0.0, value=15.0, step=1.0)
    with c3:
        q_traffic = st.selectbox("Traffic", TRAFFIC_OPTS, index=default_index(TRAFFIC_OPTS, "Medium"))
    with c4:
        q_weather = st.selectbox("Weather", WEATHER_OPTS, index=default_index(WEATHER_OPTS, "Sunny"))

    q = quick_eta_estimate(q_dist, q_prep, q_traffic, q_weather)
    glow_metric("⏱️", f"{q['total_min']} min", "Total estimated arrival")
    card_row([
        {"icon": "🛣️", "label": "Travel time", "value": f"{q['travel_min']} min",
         "sub": f"~{q['assumed_speed_kmph']} km/h assumed"},
        {"icon": "🌦️", "label": "Weather slowdown", "value": f"×{q['weather_slowdown']:g}"},
        {"icon": "🍳", "label": "Prep + handoff", "value": f"{q['prep_min'] + q['buffer_min']} min"},
    ])
    html(f'<div class="ff-formula">{q["formula"]}</div>')

    section("Versus the model", "The whole point of keeping a dumb baseline is being able to check the clever one against it.")
    model_eta = model.predict_eta(q_dist, q_traffic, 1, 1, 4.6, 30, False,
                                  "Metropolitian", q_weather, "Meal", "motorcycle")
    cmp = compare_with_model(q, model_eta)
    card_row([
        {"icon": "📐", "label": "Napkin formula", "value": f"{cmp['quick_min']} min"},
        {"icon": "🧠", "label": "Neural net", "value": f"{cmp['model_min']} min", "tone": "accent"},
        {"icon": "↔️", "label": "Gap", "value": f"{cmp['gap_min']:+g} min"},
    ])
    st.info(cmp["verdict"])
    st.caption(
        "They won't always agree, and that's expected — this formula uses the prep time you "
        "typed, while the model learned typical prep time implicitly from 45,000 real orders "
        "and can't separate it back out."
    )

# ---------------------------------------------------------------------------
# PAGE: What-If Simulator (Feature #3)
# ---------------------------------------------------------------------------
elif page == "🎛️ What-If Simulator":
    section("What-if simulator", "Drag the sliders, watch the ETA curve move live against distance.")

    w1, w2 = st.columns(2)
    with w1:
        w_traffic = st.selectbox("Traffic", TRAFFIC_OPTS,
                                 index=default_index(TRAFFIC_OPTS, "Medium"), key="w_traffic")
        w_weather = st.selectbox("Weather", WEATHER_OPTS,
                                 index=default_index(WEATHER_OPTS, "Sunny"), key="w_weather")
        w_city = st.selectbox("Area type", CITY_OPTS, key="w_city")
    with w2:
        w_multi = st.slider("Rider's concurrent orders", 0, 3, 1, key="w_multi")
        w_vcond = st.slider("Vehicle condition", 0, 3, 1, key="w_vcond")
        w_dist = st.slider("Distance (km)", DIST_MIN, DIST_MAX, DIST_DEFAULT, step=0.1, key="w_dist")
    w_festival = st.toggle("Festival day", value=False, key="w_festival")

    # one batched forward pass for the whole curve instead of 40 separate ones
    sweep = np.linspace(DIST_MIN, DIST_MAX, 40)
    curve = model.predict_many(sweep, w_traffic, w_multi, w_vcond, 4.6, 30,
                               w_festival, w_city, w_weather, "Meal", "motorcycle")
    current = model.predict_with_interval(w_dist, w_traffic, w_multi, w_vcond, 4.6, 30,
                                          w_festival, w_city, w_weather, "Meal", "motorcycle")
    half = current["high_min"] - current["eta_min"]

    fig = go.Figure()
    # the confidence band drawn as a filled ribbon — the curve alone implies a
    # precision of "27.4 minutes" that this model does not have
    fig.add_trace(go.Scatter(
        x=np.concatenate([sweep, sweep[::-1]]),
        y=np.concatenate([curve + half, (curve - half)[::-1]]),
        fill="toself", fillcolor="rgba(226,55,68,0.10)", line=dict(width=0),
        hoverinfo="skip", name="80% range",
    ))
    fig.add_trace(go.Scatter(x=sweep, y=curve, mode="lines", name="predicted ETA",
                             line=dict(width=3, color=BRAND_PRIMARY),
                             hovertemplate="%{x:.1f} km → %{y:.1f} min<extra></extra>"))
    fig.add_trace(go.Scatter(x=[w_dist], y=[current["eta_min"]], mode="markers",
                             marker=dict(size=16, symbol="star", color=BRAND_INK),
                             name="your setting",
                             hovertemplate="%{x:.1f} km → %{y:.1f} min<extra></extra>"))
    fig.update_layout(xaxis_title="Distance (km)", yaxis_title="Predicted ETA (min)",
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0))
    st.plotly_chart(style_chart(fig, 400), width="stretch")

    glow_metric(
        "⏱️", f"{current['eta_min']:,.1f} min", "ETA at your current settings",
        band=f"80% range <b>{current['low_min']:.0f}</b>–<b>{current['high_min']:.0f}</b> min",
    )
    st.caption(
        f"Distance is capped at {DIST_MIN:g}–{DIST_MAX:g} km because that's the full range the "
        f"model was trained on. Sliders read their limits from the model's own records, so a "
        f"retrain can't leave this screen offering values the model can't answer."
    )

# ---------------------------------------------------------------------------
# PAGE: Rush Hours — straight from the data, no model involved
# ---------------------------------------------------------------------------
elif page == "📊 Rush Hours":
    section(
        "When does the city slow down?",
        "Measured directly from the historical orders — no prediction involved. This is "
        "the ground truth the model was learning from.",
    )

    rush = get_rush_profile()
    peak = rush.loc[rush["avg_delay"].idxmax()]
    quiet = rush.loc[rush["avg_delay"].idxmin()]
    card_row([
        {"icon": "🐌", "label": "Slowest hour", "value": peak["label"],
         "sub": f"{peak['avg_delay']} min average", "tone": "danger"},
        {"icon": "⚡", "label": "Fastest hour", "value": quiet["label"],
         "sub": f"{quiet['avg_delay']} min average", "tone": "good"},
        {"icon": "↔️", "label": "Swing across the day",
         "value": f"{peak['avg_delay'] - quiet['avg_delay']:.1f} min", "tone": "accent"},
    ])

    fig = px.bar(rush, x="label", y="avg_delay", color="avg_delay",
                 color_continuous_scale=["#FFD9A0", "#B3261E"],
                 labels={"label": "", "avg_delay": "Average delivery (min)"},
                 hover_data={"order_count": ":,"})
    fig.update_layout(coloraxis_showscale=False)
    st.plotly_chart(style_chart(fig, 350), width="stretch")
    st.caption(
        "Worth being upfront about: hour-of-day is NOT one of the model's inputs — the "
        "training notebook never built that feature. The swing here is bigger than the "
        "effect of most things the model does use, so this is the clearest single "
        "improvement available to the next version of the model."
    )

    section("What actually costs you minutes?", "Historical averages per condition, against the overall mean.")
    impact = get_condition_impact()
    tabs = st.tabs(["🚦 Traffic", "🌦️ Weather", "🎉 Festival", "🛵 Vehicle"])
    for tab, key in zip(tabs, ["traffic", "weather", "festival", "vehicle"]):
        with tab:
            table = impact[key]
            label_col = table.columns[0]
            fig = px.bar(
                table.sort_values("vs_average_min"), x="vs_average_min", y=label_col,
                orientation="h", color="vs_average_min",
                color_continuous_scale=["#1DB954", "#F1E4DD", "#B3261E"],
                color_continuous_midpoint=0,
                labels={"vs_average_min": "Minutes vs. overall average", label_col: ""},
                hover_data={"order_count": ":,", "avg_delay": ":.1f"},
            )
            fig.update_layout(coloraxis_showscale=False)
            st.plotly_chart(style_chart(fig, 300), width="stretch")
            st.dataframe(table, width="stretch", hide_index=True)

# ---------------------------------------------------------------------------
# PAGE: Explainability + the model card
# ---------------------------------------------------------------------------
elif page == "🔍 Why This ETA?":
    section("Why did the model say that?")

    last = st.session_state.get("last_prediction")
    if not last:
        st.info("Go make a prediction in the **📦 ETA Predictor** page first, then come back here.",
                icon="👈")
    else:
        st.write(
            f"For the last prediction ({last['distance_km']} km, {last['traffic_level']} traffic, "
            f"{last['weather']} weather → **{last['pred']:.1f} min**), here's how much each "
            f"input actually mattered:"
        )
        sensitivities = model.explain(
            last["distance_km"], last["traffic_level"], last["multiple_deliveries"],
            last["vehicle_condition"], last["rider_rating"], last["rider_age"],
            last["festival"], last["city"], last["weather"],
            last["order_type"], last["vehicle_type"],
        )
        sens_df = pd.DataFrame(sensitivities.items(), columns=["feature", "influence_%"])
        fig = px.bar(sens_df.sort_values("influence_%"), x="influence_%", y="feature",
                     orientation="h", color="influence_%",
                     color_continuous_scale=["#FFD9A0", "#B3261E"],
                     labels={"influence_%": "Share of this prediction's movement (%)", "feature": ""})
        fig.update_layout(coloraxis_showscale=False)
        st.plotly_chart(style_chart(fig, 420), width="stretch")
        st.caption(
            "Continuous inputs get a ±1 standard-deviation nudge; categories get the proper "
            "counterfactual instead (\"what if this same order were in every other kind of "
            "weather?\"). Fully deterministic — the same inputs give the same explanation "
            "every time. The first version used random nudges and gave a different answer "
            "on every refresh."
        )

    st.divider()
    section("How much should you trust this?", "Every number below was measured on held-out data, not chosen.")
    card_row([
        {"icon": "🎯", "label": "Typical error", "value": f"±{CARD['mae_minutes']:.1f} min",
         "sub": f"median {CARD['medape_pct']:.1f}% off", "tone": "accent"},
        {"icon": "📈", "label": "R²", "value": f"{CARD['r2']:.3f}",
         "sub": "of the variation explained"},
        {"icon": "✅", "label": "Within 10 min", "value": f"{CARD['pct_within_10_min']:.0f}%",
         "sub": f"{CARD['pct_within_5_min']:.0f}% within 5 min", "tone": "good"},
    ])
    st.caption(
        f"Architecture {CARD['architecture']} · trained on {CARD['n_training_rows']:,} orders · "
        f"validated on {CARD['n_validation_rows']:,} it never saw during training."
    )

    with st.expander("📊 Accuracy broken down by traffic level"):
        st.caption("One blended number flatters us — we're meaningfully better in Low traffic "
                   "than in a Jam, and a dispatcher deserves to know which one they're in.")
        per_traffic = pd.DataFrame([
            {"Traffic": level, "Typical error (min)": round(stats["mae_minutes"], 2),
             "Actual average (min)": round(stats["mean_actual_minutes"], 1),
             "Orders tested": stats["n"]}
            for level, stats in CARD["per_traffic"].items()
        ])
        st.dataframe(per_traffic, width="stretch", hide_index=True)

    with st.expander("🌍 Which features the model relies on overall"):
        st.caption("Permutation importance, measured across the whole validation set during "
                   "training — how much worse the model gets when you shuffle one input. "
                   "That's the global view; the chart above is about YOUR one prediction.")
        imp = pd.DataFrame(CARD["permutation_importance"].items(),
                           columns=["feature", "importance"])
        imp = imp.sort_values("importance", ascending=False).head(12)
        fig = px.bar(imp.sort_values("importance"), x="importance", y="feature",
                     orientation="h", color_discrete_sequence=[BRAND_PRIMARY],
                     labels={"importance": "MAE increase when shuffled (min)", "feature": ""})
        st.plotly_chart(style_chart(fig, 380), width="stretch")

    with st.expander("📏 Where this model stops being reliable"):
        dom = CARD["input_domain"]
        st.markdown(
            f"""
- **Distance**: only ever saw **{dom['distance_km']['min']:.1f}–{dom['distance_km']['max']:.1f} km**.
  Anything outside gets clamped and the app says so.
- **Rider age**: **{dom['rider_age']['min']:.0f}–{dom['rider_age']['max']:.0f}**.
- **Concurrent orders**: **{dom['multiple_deliveries']['min']:.0f}–{dom['multiple_deliveries']['max']:.0f}**.
- **No hour-of-day input.** See the 📊 Rush Hours page — this is the model's biggest
  known gap, and it's a bigger effect than several features it does use.
- **The coordinates are semi-synthetic.** The Delay Hotspot Map proves the technique,
  it isn't a literal street map.
- **Prep time is baked in implicitly.** The dataset has no prep-time column, so the model
  learned an average one. It can't be told your kitchen is slower than typical today.
            """
        )

html(f'<div class="ff-footer">FeedForward · {summary["n_orders"]:,} real orders · '
     f"dataset: Kaggle 'Zomato Delivery Operations Analytics' (saurabhbadole)</div>")
