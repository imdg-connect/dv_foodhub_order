import os

from fastapi import FastAPI, HTTPException

from setup.live_dashboard import LiveFeed, http_eda

# Your EDA API (the one with POST /run-eda). It must be running on this address.
EDA_URL = os.getenv("EDA_URL", "http://127.0.0.1:8001/run-eda")

app = FastAPI(title="FoodHub Live Dashboard API")

# Every 5 s: add/update ONE row in data/foodhub_order.csv, POST EDA_URL, wait for the answer, repeat.
feed = LiveFeed(csv="data/foodhub_order.csv", eda_fn=http_eda(EDA_URL, timeout=300), interval=5)


@app.get("/")
def health():
    return {"status": "running", "live_feed": feed.running, "eda_url": EDA_URL}


@app.post("/start")
def start():
    """Start the live feed (safe to call twice: it will not start a second one)."""
    if feed.stopping:
        raise HTTPException(409, "Still finishing the current EDA after /stop. Try again in a few seconds.")
    try:
        started = feed.start()
    except (FileNotFoundError, ValueError) as e:
        raise HTTPException(400, str(e))
    return {"status": "started" if started else "already_running", **feed.status()}


@app.post("/stop")
def stop():
    """Stop the live feed. If an EDA run is in progress it is allowed to finish first."""
    if not feed.running:
        return {"status": "not_running", **feed.status()}
    feed.stop(wait=2)                               # do not hang the request while the EDA finishes
    return {"status": "stopping" if feed.running else "stopped", **feed.status()}


@app.get("/status")
def status():
    return feed.status()