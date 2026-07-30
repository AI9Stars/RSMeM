#!/usr/bin/env bash
# Launch Perception.py as an MCP stdio server (same interpreter as the agent CLI).
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
exec python3 "$SCRIPT_DIR/Perception.py" "$@"
