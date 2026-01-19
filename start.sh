#!/bin/bash
# start.sh - Start CT Viewer on Mac/Linux
# Automatically creates a virtual environment if needed

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PORT=7002
URL="http://localhost:$PORT"
VENV_DIR="$SCRIPT_DIR/venv"

echo ""
echo "========================================"
echo "         CT VIEWER FOR MAC"
echo "========================================"
echo ""

# Check if Python 3 is available
if ! command -v python3 &> /dev/null; then
    echo " ERROR: Python 3 is not installed."
    echo ""
    echo " Please install Python from:"
    echo " https://www.python.org/downloads/"
    echo ""
    echo " Press Enter to exit..."
    read
    exit 1
fi

echo " Python: $(python3 --version)"

# Check for bundled environment first
if [ -f "./ctviewer_env/bin/python" ]; then
    PYTHON_CMD="./ctviewer_env/bin/python"
    echo " Using bundled ctviewer_env"
elif [ -f "$VENV_DIR/bin/python" ]; then
    PYTHON_CMD="$VENV_DIR/bin/python"
    echo " Using local virtual environment"
else
    # Create virtual environment
    echo ""
    echo " Setting up virtual environment (first time only)..."
    python3 -m venv "$VENV_DIR"

    if [ ! -f "$VENV_DIR/bin/python" ]; then
        echo " ERROR: Failed to create virtual environment."
        echo " Press Enter to exit..."
        read
        exit 1
    fi

    PYTHON_CMD="$VENV_DIR/bin/python"

    echo " Installing required packages..."
    "$VENV_DIR"/bin/pip install --upgrade pip > /dev/null 2>&1
    "$VENV_DIR"/bin/pip install flask pillow numpy scipy scikit-image pydicom pylibjpeg pylibjpeg-libjpeg

    if [ $? -ne 0 ]; then
        echo ""
        echo " ERROR: Failed to install packages."
        echo " Press Enter to exit..."
        read
        exit 1
    fi

    echo ""
    echo " Setup complete!"
fi

echo ""

# Kill any existing process on the port
EXISTING_PID=$(lsof -ti:$PORT 2>/dev/null)
if [ -n "$EXISTING_PID" ]; then
    echo " Stopping existing server on port $PORT..."
    kill -9 $EXISTING_PID 2>/dev/null
    sleep 2
fi

echo " Starting CT Viewer..."
echo ""
echo " ----------------------------------------"
echo " To STOP the viewer: Close this window"
echo "                or press Ctrl+C"
echo " ----------------------------------------"
echo ""

# Start server in background
"$PYTHON_CMD" ct_viewer.py &
SERVER_PID=$!

# Wait for server to be ready
echo " Waiting for server..."
for i in {1..30}; do
    if curl -s "$URL" > /dev/null 2>&1; then
        echo " Server is ready!"
        break
    fi
    sleep 1
done

# Open browser
echo " Opening browser to: $URL"
if [[ "$OSTYPE" == "darwin"* ]]; then
    open "$URL"
elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
    xdg-open "$URL" 2>/dev/null || echo " Please open $URL in your browser"
fi

# Wait for server
wait $SERVER_PID
