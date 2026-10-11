"""
agents_api.py - thin entry point on top of agents/agents_hub.py

Run options (from the project root, FOODHUB_...):
  1) CLI :  python api/agents_api.py "Which cuisine earns the most revenue?"
  2) REST:  uvicorn api.agents_api:app --reload      then open http://127.0.0.1:8000/docs
"""

import asyncio
import sqlite3
import sys
from pathlib import Path

# STEP 1: make the project root importable so "agents.agents_hub" resolves
sys.path.append(str(Path(__file__).resolve().parents[1]))

from agents.agents_hub import get_ctx, read_schema, run_pipeline   # STEP 2: import the pipeline

# ---------------------------------------------------------------------------
# STEP 3: REST API (FastAPI)
# ---------------------------------------------------------------------------
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from pydantic import BaseModel

app = FastAPI(title="FoodHub Multi-Agent API")


class Question(BaseModel):          # request body: {"question": "..."}
    question: str


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/schema")
def schema():
    return read_schema()


@app.post("/ask")
def ask(body: Question):
    if not body.question.strip():
        raise HTTPException(status_code=400, detail="Question is empty")
    return run_pipeline(body.question)   # runs Layers 0 -> 4


# ---------------------------------------------------------------------------
# STEP 3b: dashboard data straight from the same SQLite DB (read-only)
# ---------------------------------------------------------------------------
def fetch_orders() -> list:
    ctx = get_ctx()                                   # same db_path + main table the agents use
    conn = sqlite3.connect(f"file:{ctx['db_path']}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    try:
        return [dict(r) for r in conn.execute(f"SELECT * FROM {ctx['table']}")]
    finally:
        conn.close()


@app.get("/orders")
def orders():
    try:
        return fetch_orders()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Could not read orders from SQLite: {e}")


@app.websocket("/ws")
async def orders_ws(websocket: WebSocket):
    """Pushes all rows on connect, then again every time the database changes."""
    await websocket.accept()
    watcher = None
    try:
        ctx = get_ctx()
        watcher = sqlite3.connect(f"file:{ctx['db_path']}?mode=ro", uri=True)
        version = None
        while True:
            v = watcher.execute("PRAGMA data_version").fetchone()[0]   # changes when another connection commits
            if v != version:
                version = v
                await websocket.send_json({"type": "orders", "rows": fetch_orders()})
            await asyncio.sleep(1)
    except WebSocketDisconnect:
        pass
    except Exception as e:
        try:
            await websocket.send_json({"type": "error", "detail": str(e)})
        except Exception:
            pass
    finally:
        if watcher:
            watcher.close()


# ---------------------------------------------------------------------------
# STEP 4: CLI mode
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    q = " ".join(sys.argv[1:]) or "Which cuisines earn the most revenue and how fast is delivery?"
    result = run_pipeline(q)
    print("\nAGENTS USED:", result["agents"])
    print("\n=== EXECUTIVE REPORT ===\n" + result["report"])
    print("\n=== TRACE ===")
    for step in result["trace"]:
        print(step)