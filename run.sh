#!/usr/bin/env bash
# One command: install, build the interface, serve everything on http://localhost:8000
set -euo pipefail
cd "$(dirname "$0")"
command -v ffmpeg >/dev/null || echo "warning: FFmpeg not found; MP3 export and some uploads won't work"
[ -f .env ] || echo "note: no .env found; running in demo mode (copy .env.example to .env to add keys)"
# Use a project virtualenv (system Pythons often refuse global pip installs, PEP 668)
if [ -z "${VIRTUAL_ENV:-}" ]; then
  [ -d .venv ] || python3 -m venv .venv
  # shellcheck disable=SC1091
  source .venv/bin/activate
fi
python -m pip install -q --upgrade pip
python -m pip install -q -r backend/requirements.txt
(cd frontend && npm install --no-audit --no-fund --silent && npm run build --silent)
cd backend && exec python -m uvicorn app.main:app --host 127.0.0.1 --port "${PORT:-8000}"
