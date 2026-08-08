# 🍔➡️ FeedForward — Smart Delivery ETA

"FeedForward" is a pun on purpose — food delivery ("feed") + the actual
technical name of the neural network we used (a **feedforward network**,
the plain Dense-layer kind, no loops/attention/anything fancy).

Built for a hackathon. Runs fully offline once installed — no paid APIs,
no API keys, nothing that needs a subscription.

---

## The problem

Food delivery ETAs today mostly optimize one thing: get the customer's
countdown number as low as possible. Two side effects of that get ignored:

1. **Riders get stacked** with concurrent orders to hit those numbers,
   which pushes people to rush in traffic they shouldn't be rushing in —
   a real, documented gig-economy labor/safety problem.
2. **Some zones are just worse served** than others, and "traffic is bad"
   is treated as background noise instead of a pattern worth surfacing to
   whoever plans rider coverage.

FeedForward is an ETA predictor that also flags both of those, instead of
pretending the only number that matters is speed.

## Features

| # | Feature | What it does |
|---|---------|---------------|
| 1 | 🗺️ **Delay Hotspot Map** | Real restaurant GPS points grouped into zones and **named by city**, coloured by average delivery time — surfaces underserved areas instead of burying them in an average. |
| 2 | 🛒 **Order Food** | A working customer flow: pick a restaurant, build a cart, set conditions, place a mock order and track it. Real model output, not a scripted animation. |
| 3 | 🏪 **Restaurant Hub** | The other side: accept orders, get a rider dispatched automatically to whoever is **least loaded**, and see the whole fleet's fairness picture. |
| 4 | 📦 **ML ETA Predictor** | Neural net trained on 45,584 real Zomato delivery records, with a confidence band and the Rider Fairness Score bundled in so nobody can skip it. |
| 5 | 🍽️ **Quick Calculator** | No-ML baseline: assumed speed per traffic level + weather slowdown + prep time. Shows its own formula, and compares itself against the model. |
| 6 | 🎛️ **What-If Simulator** | Live sliders, real-time ETA-vs-distance curve with its uncertainty ribbon. |
| 7 | 📊 **Rush Hours** | Straight from the data, no prediction: when the city actually slows down, and what each condition really costs you in minutes. |
| 8 | 🔍 **Why This ETA?** | Deterministic local explainability, plus the model's own measured accuracy and an explicit list of where it stops being reliable. |

## Dataset

[Zomato Delivery Operations Analytics Dataset](https://www.kaggle.com/datasets/saurabhbadole/zomato-delivery-operations-analytics-dataset)
(Kaggle, by saurabhbadole) — 45,584 real delivery orders. Columns include
rider age/rating, restaurant + delivery GPS coordinates, weather, traffic
density, vehicle type/condition, concurrent-order count, festival flag,
area type (city tier), and the actual delivery time. Already included in
`data/food_delivery.csv`, works out of the box.

Two things we found in it that weren't obvious:

- **The city codes.** Every `Delivery_person_ID` looks like `JAPRES09DEL03`,
  and the part before `RES` is a real city code — 22 of them. We confirmed
  it by taking each code's median coordinate and checking it landed on the
  city we'd guessed (JAP → 26.91N 75.79E → Jaipur). So the map names real
  Indian cities instead of labelling zones by grid coordinate.
- **Null island.** 3,640 rows have their coordinates recorded as exactly
  `0.0` — a missing value someone wrote as a zero. They all bucketed into
  one grid cell at (0, 0), which is in the Gulf of Guinea off West Africa,
  and it rendered as the *second-largest delivery hotspot on the map*. It
  looked completely plausible until you checked where the dot actually was.

## How the model works (short version)

A small Keras MLP (25 → 32 → 16 → 1) trained on the dataset, predicting
`Time_taken (min)`. Full training + EDA + explainability writeup — including
a genuinely funny GPS coordinate bug we hit and fixed — is in
[`model/train_model.ipynb`](model/train_model.ipynb).

Measured on 6,838 held-out orders it never saw during training:

| Metric | Value |
|---|---|
| Typical error (MAE) | **3.59 minutes** |
| Median error | 11.7% |
| R² | 0.766 |
| Within 5 / 10 minutes | 72.9% / 97.1% |

Accuracy varies by conditions, so the app shows the figure for whichever
one you picked rather than a single blended number — we're within 3.1 min
in Low traffic and 4.0 min in a Jam.

Same trick as BeejBank: **the app never imports TensorFlow.** After
training, the learned weights are exported to `weights.npz` and the
forward pass is reimplemented in a few lines of numpy
(`backend/inference.py`) — no need to ship a 300MB+ framework just to
replay a 3-layer network.

## Project structure

```
feedforward/
├── frontend/
│   ├── app.py                # the Streamlit UI (all 8 screens)
│   └── assets/style.css      # the whole custom skin, one file
├── backend/
│   ├── inference.py          # numpy-only forward pass, input guarding,
│   │                         #   confidence bands, explainability
│   ├── quick_calc.py         # no-ML baseline ETA formula
│   ├── fairness.py           # rider load scoring + the fleet view
│   ├── marketplace.py        # the mock two-sided ordering flow + dispatch
│   └── data_utils.py         # dataset loading, map + analytics views
├── model/
│   ├── train_model.ipynb     # training notebook (run this to retrain)
│   ├── weights.npz           # exported model weights (numpy-loadable)
│   ├── meta.json             # encodings, scaler stats, measured accuracy,
│   │                         #   and the input domain the app clamps to
│   └── architecture.png      # model architecture diagram (visualkeras)
├── tests/                    # 158 checks, ~27 seconds, incl. a headless
│                             #   run of the real app through every screen
├── data/food_delivery.csv    # the Kaggle dataset
├── ppt/presentation.pptx     # pitch deck
├── requirements.txt          # runtime deps (what run.bat installs)
├── requirements-dev.txt      # + deps only needed to retrain or run tests
└── run.bat / run.sh          # one-click launchers
```

## Running it

### Windows (portable, plug-n-play)

Unzip the folder anywhere, double-click **`run.bat`**. First run sets up a
local virtual environment and installs the (small) dependency list from
`requirements.txt`, then opens the app in your browser. Needs Python 3.10+
on the machine and one-time internet access to fetch the 4 packages above —
after that first run it works offline.

### Linux / Mac (dev)

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
streamlit run frontend/app.py
```

### Running the tests

```bash
pip install -r requirements-dev.txt
pytest
```
→ 158 tests, about 27 seconds. They cover every backend module and boot the
real Streamlit app headlessly through all 8 screens, clicking the full
order → accept → dispatch → delivered flow. Several are labelled
`REGRESSION` because they exist specifically so a bug we already shipped
can't come back.

### Retraining the model

```bash
pip install -r requirements-dev.txt
jupyter nbconvert --to notebook --execute --inplace model/train_model.ipynb
```
This regenerates `weights.npz`, `meta.json` and `architecture.png` together.
The app picks up the new brain next time it starts — and if the two files
end up mismatched, it refuses to start rather than predicting garbage.

## What we got wrong and fixed

The interesting part of this project, honestly. All of these shipped in an
earlier version and all of them have a regression test now.

1. **Both landing-page buttons crashed.** "I'm hungry" and "I run a kitchen"
   wrote to the navigation radio's own `session_state` key *after* that
   widget had been created, which Streamlit forbids. The two most prominent
   buttons in the app raised an exception instead of navigating.
2. **The fairness score was fed a dice roll.** Rider assignment was
   `random.choice(RIDERS)`, and how many orders that rider was already
   carrying was `random.choice([0, 1, 2])` — so the one feature this whole
   project is pitched on was computed from randomness rather than from what
   was happening in the app. Riders now carry their real workload, and
   dispatch hands each order to the least-loaded one.
3. **The customer's conditions were thrown away.** The "simulate conditions"
   controls were collected on the order screen and then silently discarded;
   the restaurant screen picked its own values. The festival toggle was
   hardcoded to `False` on the way into the model.
4. **Null island on the map** — see the Dataset section above.
5. **The glass login card was empty.** It was a `<div>` opened in one
   `st.markdown()` and closed in another. Streamlit renders each markdown
   block as its own isolated node and auto-closes hanging tags, so the card
   rendered as an empty box next to a completely unstyled login form.
6. **Explainability changed its answer on every refresh.** It nudged every
   input with a random gaussian — including the 18 one-hot columns, which
   put the model in states like "0.6 Foggy" that it has never seen, so those
   columns scored high purely for being pushed somewhere nonsensical. It's
   deterministic now, and categories get a proper counterfactual instead.
7. **Nothing stopped you asking about a 100 km delivery.** The distance box
   was unbounded, but the model has only ever seen 1.5–21 km. Inputs are
   clamped to the model's recorded training domain now, and the app says
   when it clamped.
8. **Raw HTML printed in a grey code box on every single screen.** Streamlit
   runs our markup through a Markdown parser even with
   `unsafe_allow_html=True`, and Markdown treats any line indented four or
   more spaces as a *code block*. Our top bar was a neatly-indented f-string
   with a conditional "N on the road" chip — and whenever nobody had an order
   on the road, that chip collapsed to `""`, leaving a whitespace-only line
   that ended the HTML block. Everything after it rendered as literal source.
   The headless tests never saw it: the element was present, it just wasn't
   *rendering* as HTML. It took opening the app in a real browser. All raw
   markup now goes through one helper that flattens it to a single line.

## Known limitations (being upfront about it)

- **No hour-of-day input.** The training notebook never built that feature,
  and the 📊 Rush Hours screen shows why that's a real gap: the gap between
  the quietest hour and the dinner rush is about 12 minutes, larger than the
  effect of several features the model *does* use. It's the clearest single
  improvement available to the next version.
- **The Quick Calculator and the model diverge with distance, a lot.** The
  napkin formula scales travel time linearly at an assumed 25 km/h; the
  model barely moves with distance at all (about 4 minutes across its whole
  1.5–21 km range), because in this dataset distance is a genuinely weak
  predictor. By 15 km the formula is half an hour more pessimistic. That's
  the comparison feature doing its job — but treat the formula as the more
  trustworthy one for long routes, since the model has very little real
  signal there. There's a test pinning this so it can't drift silently.
- **The GPS coordinates are semi-synthetic** — see the notebook for the
  sign-error bug we found and fixed. The Delay Hotspot Map is a
  proof-of-concept of the *technique*, not a literal street map.
- **Prep time is baked in implicitly.** The dataset has no prep-time column,
  so the ML model's prediction already includes a typical historical prep
  time and can't be told your kitchen is slow today — unlike the Quick
  Calculator, which asks you for it directly.
- **The Rider Fairness Score is a simple, auditable weighted formula on
  purpose**, not a second ML model — a safety/fairness flag should be easy
  to explain, not another black box stacked on the first one.
- **The marketplace screens are a simulation.** The restaurants, menus and
  test riders are demo content. Every ETA and fairness number they produce
  is real model output, but nobody is actually delivering you a pizza.

## Team / hackathon

Built by Sagnik, Indrashish, Nilarko, Rudra, Adwitiya and Urbi.
See [`TEAM_GUIDE.txt`](TEAM_GUIDE.txt) for who built what and why.
