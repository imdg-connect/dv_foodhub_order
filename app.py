"""
app.py - single entry point for Hugging Face Spaces (one port: 7860).

It mounts your three existing FastAPI apps unchanged and serves the built React UI:

    /api/*   -> api/agents_api.py     (/api/ask, /api/orders, /api/ws ...)
    /dash/*  -> api/dashboard_api.py  (/dash/start, /dash/stop, /dash/status)
    /eda/*   -> api/eda_api.py        (/eda/run-eda)
    /        -> foodhub-ui/dist       (React build)

Run locally:  npm --prefix foodhub-ui run build   then   uvicorn app:app --port 7860
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

# ---- ZeroGPU compatibility -------------------------------------------------------------------------------
# ZeroGPU Spaces refuse to start unless a @spaces.GPU function is wired into a Gradio app. This app is CPU-only,
# so we add a tiny hidden Gradio page (served at /gradio) whose button calls a placeholder GPU function.
# ZeroGPU allocates a GPU only when that function is called, so it costs nothing. On any other host
# (your PC, Render, ...) this just becomes a harmless extra page, or is skipped if gradio is not installed.
try:
    import gradio as gr

    try:
        import spaces
        _gpu = spaces.GPU(duration=1)
    except ImportError:                              # not on ZeroGPU: plain function
        _gpu = lambda f: f                           # noqa: E731

    @_gpu
    def _zerogpu_probe() -> str:
        return "ok"

    with gr.Blocks() as _probe_ui:
        gr.Markdown("FoodHub: ZeroGPU probe page (unused). The app is at the site root.")
        _btn = gr.Button("probe")
        _out = gr.Textbox()
        _btn.click(_zerogpu_probe, None, _out)
    HAVE_GRADIO = True
except ImportError:
    HAVE_GRADIO = False

app = FastAPI(title="FoodHub Command Center")


@app.get("/healthz")
def healthz():
    return {"ok": True}


app.mount("/api", agents_app)
app.mount("/dash", dashboard_app)
app.mount("/eda", eda_app)

if HAVE_GRADIO:                                    # must come BEFORE the static "/" mount below
    app = gr.mount_gradio_app(app, _probe_ui, path="/gradio", ssr_mode=False)

DIST = ROOT / "foodhub-ui" / "dist"
if DIST.exists():                                  # mounted last so it never shadows the API routes
    app.mount("/", StaticFiles(directory=DIST, html=True), name="ui")


def _bootstrap():
    """A Space's disk is wiped on restart, so rebuild processed/foodhub.db once the server is up."""
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


if __name__ == "__main__":                         # Gradio-SDK Spaces start the app with: python app.py
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=PORT)