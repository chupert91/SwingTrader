"""Vercel ASGI entrypoint.

Vercel's Python framework preset detects FastAPI from requirements.txt and
looks for a top-level `app` variable in app.py/main.py/index.py (also in
src/, app/, api/). This file just re-exports the real app defined in
backend/main.py so Vercel finds it without us having to move modules.

For local dev, `python run.py` continues to import backend.main directly.
"""
from backend.main import app  # noqa: F401
