#!/bin/bash
# restart.sh - Restart CT Viewer server

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "Restarting CT Viewer..."
echo ""

# Stop existing server
"$SCRIPT_DIR/stop.sh"

# Wait a moment for port to be released
sleep 1

# Start server
"$SCRIPT_DIR/start.sh"
