import os
import threading

from fastapi import FastAPI, HTTPException

from setup.live_dashboard import LiveFeed, http_eda

# Your EDA API (the one with POST /run-eda). It must be running on this address.
EDA_URL = os.getenv("EDA_URL", "http://127.0.0.1:8001/run-eda")

# The live feed switches itself off this many seconds after a NEW run starts.
AUTO_OFF_S = int(os.getenv("LIVE_AUTO_OFF_SECONDS", "30"))

app = FastAPI(title="FoodHub Live Dashboard API")

# Every 5 s: add/update ONE row in data/foodhub_order.csv, POST EDA_URL, wait for the answer, repeat.
feed = LiveFeed(csv="data/foodhub_order.csv", eda_fn=http_eda(EDA_URL, timeout=300), interval=5)

_timer = None
_timer_lock = threading.Lock()


def _auto_stop():
    """Fires AUTO_OFF_S seconds after start: same effect as POST /stop."""
    if feed.running:
        feed.stop(wait=2)          # do not hang while the last EDA finishes


def _cancel_timer():
    global _timer
    if _timer:
        _timer.cancel()
        _timer = None


@app.get("/")
def health():
    return {"status": "running", "live_feed": feed.running, "eda_url": EDA_URL, "auto_off_seconds": AUTO_OFF_S}


@app.post("/start")
def start():
    """Start the live feed (safe to call twice: it will not start a second one). Auto-stops after AUTO_OFF_S."""
    global _timer
    if feed.stopping:
        raise HTTPException(409, "Still finishing the current EDA after /stop. Try again in a few seconds.")
    try:
        started = feed.start()
    except (FileNotFoundError, ValueError) as e:
        raise HTTPException(400, str(e))
    if started:                                    # only a NEW run arms the timer, so extra clicks never extend it
        with _timer_lock:
            _cancel_timer()
            _timer = threading.Timer(AUTO_OFF_S, _auto_stop)
            _timer.daemon = True
            _timer.start()
    return {"status": "started" if started else "already_running", "auto_off_seconds": AUTO_OFF_S, **feed.status()}


@app.post("/stop")
def stop():
    """Stop the live feed. If an EDA run is in progress it is allowed to finish first."""
    with _timer_lock:                              # a manual stop cancels the pending auto-stop
        _cancel_timer()
    if not feed.running:
        return {"status": "not_running", **feed.status()}
    feed.stop(wait=2)                              # do not hang the request while the EDA finishes
    return {"status": "stopping" if feed.running else "stopped", **feed.status()}


@app.get("/status")
def status():
    return {**feed.status(), "auto_off_seconds": AUTO_OFF_S}