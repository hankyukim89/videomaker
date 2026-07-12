#!/bin/bash
# AI Video Maker – Mac launcher
# Double-click this file in Finder to start the app.

# Change to the folder this script lives in
cd "$(dirname "$0")"

# ── Check for Python 3 ──────────────────────────────────────────────────────
PY=""
for candidate in python3 python; do
  if command -v "$candidate" &>/dev/null; then
    ver=$("$candidate" -c "import sys; print(sys.version_info.major)" 2>/dev/null)
    if [ "$ver" = "3" ]; then
      PY="$candidate"
      break
    fi
  fi
done

if [ -z "$PY" ]; then
  echo "Python 3 was not found."
  echo "Install it from https://www.python.org/downloads/ and run this again."
  read -r -p "Press Enter to close..."
  exit 1
fi

# ── Create virtual environment on first run ──────────────────────────────────
if [ ! -d ".venv" ]; then
  echo "First run: setting up environment (this takes a minute or two)..."
  "$PY" -m venv .venv
  if [ $? -ne 0 ]; then
    echo "Failed to create virtual environment."
    read -r -p "Press Enter to close..."
    exit 1
  fi
fi

# ── Install / update dependencies ────────────────────────────────────────────
echo "Checking dependencies..."
.venv/bin/pip install -q -r requirements.txt
if [ $? -ne 0 ]; then
  echo "Dependency installation failed – see above."
  read -r -p "Press Enter to close..."
  exit 1
fi

# ── Open browser after a short delay so the server has time to start ─────────
(sleep 2 && open "http://127.0.0.1:8765") &

echo ""
echo "Starting AI Video Maker at http://127.0.0.1:8765"
echo "Keep this window open while you use the app. Close it to stop."
echo ""

# ── Start the server ──────────────────────────────────────────────────────────
.venv/bin/python server.py

echo ""
echo "Server stopped."
read -r -p "Press Enter to close..."
