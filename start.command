#!/usr/bin/env bash
cd "$(dirname "$0")"

echo "==========================================="
echo "🧠 Starting Brain Data Explorer Dashboard"
echo "==========================================="

if [ ! -d ".venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv .venv
    source .venv/bin/activate
    pip install -r requirements.txt
else
    source .venv/bin/activate
fi

echo "Launching Streamlit..."
streamlit run app.py
