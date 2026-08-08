#!/usr/bin/env bash
# Same idea as run.bat, for Mac/Linux — useful for practicing before the
# hackathon or if a judge happens to be on a non-Windows laptop.
set -e
cd "$(dirname "$0")"

echo
echo " FeedForward - Smart Delivery ETA"
echo " ----------------------------------"
echo

if [ ! -f ".venv/bin/python" ]; then
    echo "[1/3] First time setup - creating a local Python toolbox, one moment..."
    python3 -m venv .venv
fi

if [ ! -f ".venv/feedforward_installed.marker" ]; then
    echo "[2/3] Installing app requirements (needs internet, only happens once)..."
    .venv/bin/python -m pip install --quiet --upgrade pip
    .venv/bin/python -m pip install --quiet -r requirements.txt
    echo done > ".venv/feedforward_installed.marker"
else
    echo "[2/3] Requirements already installed, skipping."
fi

mkdir -p "$HOME/.streamlit"
if [ ! -f "$HOME/.streamlit/credentials.toml" ]; then
    printf '[general]\nemail = ""\n' > "$HOME/.streamlit/credentials.toml"
fi

echo "[3/3] Starting FeedForward... a browser tab will open automatically."
echo "       (Ctrl+C to stop the app)"
echo
.venv/bin/python -m streamlit run "frontend/app.py" --browser.gatherUsageStats false
