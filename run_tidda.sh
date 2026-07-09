#!/bin/bash
# TIDDA C2 Launcher for Linux

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="$ROOT/.venv"
PYTHON="$VENV/bin/python"
SERVER="$ROOT/tidda_lightweight_swarm.py"
FRONTEND="$ROOT/html/TIDDA_GCS_real h ai wala.html"
PORT=8000

echo "============================================"
echo "  TIDDA C2 - Linux Launcher"
echo "============================================"
echo ""
echo "  Project : $ROOT"
echo "  Python  : $PYTHON"
echo "  Server  : $SERVER"
echo "  Frontend: $FRONTEND"
echo ""

# Check for venv python
if [ ! -f "$PYTHON" ]; then
    echo "[ERROR] Python not found in virtual environment at:"
    echo "        $PYTHON"
    echo "Please create a virtual environment first:"
    echo "  python3 -m venv .venv"
    exit 1
fi
echo "[OK] Found venv python."

# Check for server script
if [ ! -f "$SERVER" ]; then
    echo "[ERROR] Backend server not found at: $SERVER"
    exit 1
fi
echo "[OK] Found server script."

# Check for port conflict on 8000
echo "  Checking port $PORT..."
if command -v lsof > /dev/null; then
    PID=$(lsof -t -i:$PORT)
    if [ ! -z "$PID" ]; then
        echo "[WARNING] Port $PORT is already in use by process PID $PID"
        read -p "  Kill existing process? (y/n): " KILL_CHOICE
        if [[ "$KILL_CHOICE" =~ ^[Yy]$ ]]; then
            kill -9 $PID
            echo "  [OK] Killed process $PID"
            sleep 2
        else
            echo "  [ABORT] Port $PORT is busy."
            exit 1
        fi
    fi
fi
echo "[OK] Port $PORT is free."

# Open frontend in default browser
if [ -f "$FRONTEND" ]; then
    echo "  Opening GCS Frontend in browser..."
    if command -v xdg-open > /dev/null; then
        xdg-open "$FRONTEND" &
        echo "[OK] Frontend launched."
    elif command -v gnome-open > /dev/null; then
        gnome-open "$FRONTEND" &
        echo "[OK] Frontend launched."
    else
        echo "[WARNING] Could not open browser automatically. Please open the HTML file manually in your browser:"
        echo "          $FRONTEND"
    fi
else
    echo "[WARNING] Frontend HTML not found."
fi

# Run server
echo ""
echo "============================================"
echo "  Starting TIDDA C2 Server on port $PORT"
echo "  Press Ctrl+C to stop the server."
echo "============================================"
source "$VENV/bin/activate"
python "$SERVER"
