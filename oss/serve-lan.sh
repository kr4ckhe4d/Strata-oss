#!/usr/bin/env bash
# oss/serve-lan.sh - serve gpt-oss-120b on the network (0.0.0.0) with Strata's dashboard and the EAGLE3 draft.
#
#   oss/serve-lan.sh            start in the background (log: oss-server.log in the repo)
#   oss/serve-lan.sh stop       stop it (the engine saves its expert profile on the way out)
#   oss/serve-lan.sh status     is it running, and its health
#
# Listening beyond 127.0.0.1 needs an API key: the same one as the Qwen Strata server, read from its key file
# (STRATA_OSS_KEY_FILE to use another). Environment overrides: MODEL_DIR, PORT, CTX, PARALLEL.
# Stop the Qwen server and llama-server first: this needs both GPUs and most of the RAM.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL_DIR="${MODEL_DIR:-/mnt/fast/models/gpt-oss-120b}"
PORT="${PORT:-8096}"
CTX="${CTX:-131072}"
PARALLEL="${PARALLEL:-1}"
KEY_FILE="${STRATA_OSS_KEY_FILE:-$HOME/code/Strata/.strata-api-key}"
LOG="$REPO/oss-server.log"

running() { pgrep -f "oss/server.py --model" >/dev/null; }

case "${1:-start}" in
stop)
    if ! running; then echo "not running"; exit 0; fi
    pkill -TERM -f "oss/server.py --model"
    for _ in $(seq 1 120); do pgrep -f "build-oss/strata-oss --model" >/dev/null || break; sleep 1; done
    echo "stopped"
    exit 0
    ;;
status)
    if running; then
        echo "running: http://$(hostname -I | awk '{print $1}'):$PORT/"
        curl -s -m 3 "localhost:$PORT/health" && echo
    else
        echo "not running"
    fi
    exit 0
    ;;
start) ;;
*) echo "usage: $0 [start|stop|status]"; exit 2 ;;
esac

if running; then echo "already running (oss/serve-lan.sh stop first)"; exit 1; fi
if [ ! -s "$KEY_FILE" ]; then echo "no API key file at $KEY_FILE (set STRATA_OSS_KEY_FILE)"; exit 1; fi
for f in "$REPO/build-oss/strata-oss" "$MODEL_DIR/gpt-oss-120b-MXFP4.gguf"; do
    [ -e "$f" ] || { echo "missing: $f"; exit 1; }
done
if pgrep -f "serve/server.py --engine strata" >/dev/null || pgrep -f "llama-server" >/dev/null; then
    echo "the Qwen Strata server or llama-server is running: stop them first (they need the same GPUs and RAM)"
    exit 1
fi

DRAFT=()
[ -e "$MODEL_DIR/eagle3-gpt-oss-120b-Q8_0.gguf" ] && DRAFT=(--draft-model "$MODEL_DIR/eagle3-gpt-oss-120b-Q8_0.gguf")

cd "$REPO"
setsid nohup .venv/bin/python oss/server.py --model "$MODEL_DIR/gpt-oss-120b-MXFP4.gguf" "${DRAFT[@]}" \
    --name gpt-oss-120b --host 0.0.0.0 --port "$PORT" --api-key "$(tr -d '\n' < "$KEY_FILE")" \
    -- --ctx "$CTX" --parallel "$PARALLEL" --profile "$MODEL_DIR/oss120.prof" > "$LOG" 2>&1 < /dev/null &
disown

echo -n "loading (~20 s)"
for _ in $(seq 1 120); do
    curl -s -m 2 "localhost:$PORT/health" >/dev/null 2>&1 && break
    running || { echo; echo "failed to start; see $LOG"; tail -5 "$LOG"; exit 1; }
    echo -n "."; sleep 2
done
echo
echo "dashboard: http://$(hostname -I | awk '{print $1}'):$PORT/   (API key: the one in $KEY_FILE)"
echo "API:       http://$(hostname -I | awk '{print $1}'):$PORT/v1"
echo "log:       $LOG"
