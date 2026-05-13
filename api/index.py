"""Vercel ASGI entrypoint.

Lives at api/index.py because Vercel's `functions` config only accepts
paths inside the api/ directory. A catch-all rewrite in vercel.json
('/(.*)' -> '/api/index') sends every request through this file, where
the FastAPI app handles both /api/* routes and the static frontend
(via app.mount('/', StaticFiles(...)) at the bottom of backend/main.py).

For local dev, `python run.py` continues to import backend.main directly.
"""
import sys
from pathlib import Path

# Ensure the project root is on sys.path so `backend` resolves on Vercel.
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from backend.main import app  # noqa: F401,E402
