"""
app.py - single entry point (FastAPI) for Render / local use.

Mounts the three FastAPI apps and serves the built React UI:

    /api/*   -> api/agents_api.py     (/api/ask, /api/orders, /api/ws ...)
    /dash/*  -> api/dashboard_api.py  (/dash/start, /dash/stop, /dash/status)
    /eda/*   -> api/eda_api.py        (/eda/run-eda)
    /        -> foodhub-ui/dist       (React build)

Run locally:  npm --prefix foodhub-ui run build   then   uvicorn app:app --port 7860
Render start: uvicorn app:app --host 0.0.0.0 --port $PORT
"""
import os
import threading
import time
from pathlib import Path

import requests
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

ROOT = Path(__file__).resolve().parent
os.chdir(ROOT)                                    # dashboard_api uses the relative path data/foodhub_order.csv
PORT = int(os.getenv("PORT", "7860"))

# must be set BEFORE the api modules are imported (they read these at import time)
os.environ.setdefault("EDA_URL", f"http://127.0.0.1:{PORT}/eda/run-eda")
os.environ.setdefault("USE_EMBEDDINGS", "0")      # skip the heavy sentence-transformers model

from api.agents_api import app as agents_app        # noqa: E402
from api.dashboard_api import app as dashboard_app  # noqa: E402
from api.eda_api import app as eda_app              # noqa: E402

app = FastAPI(title="FoodHub Command Center")


@app.get("/healthz")
def healthz():
    return {"ok": True}


app.mount("/api", agents_app)
app.mount("/dash", dashboard_app)
app.mount("/eda", eda_app)

DIST = ROOT / "foodhub-ui" / "dist"
if DIST.exists():                                  # mounted last so it never shadows the API routes
    app.mount("/", StaticFiles(directory=DIST, html=True), name="ui")


def _bootstrap():
    """The disk is wiped on every restart/redeploy, so rebuild processed/foodhub.db once the server is up."""
    if (ROOT / "processed" / "foodhub.db").exists():
        return
    for _ in range(60):                            # wait for the server to accept requests
        try:
            if requests.get(f"http://127.0.0.1:{PORT}/healthz", timeout=2).ok:
                break
        except requests.RequestException:
            pass
        time.sleep(1)
    else:
        return
    try:
        r = requests.post(os.environ["EDA_URL"], timeout=600)
        print(f"[bootstrap] EDA -> HTTP {r.status_code}", flush=True)
    except Exception as e:
        print(f"[bootstrap] EDA failed: {e}", flush=True)


@app.on_event("startup")
def _startup():
    threading.Thread(target=_bootstrap, daemon=True, name="bootstrap").start()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=PORT)