#!/bin/bash
# stop.sh - Stop CT Viewer server

PORT=7002

EXISTING_PID=$(lsof -ti:$PORT 2>/dev/null)
if [ -n "$EXISTING_PID" ]; then
    echo "Stopping CT Viewer on port $PORT (PID: $EXISTING_PID)..."
    kill -9 $EXISTING_PID 2>/dev/null
    echo "Server stopped."
else
    echo "No CT Viewer running on port $PORT."
fi
