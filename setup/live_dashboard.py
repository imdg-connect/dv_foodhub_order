"""
live_dashboard.py - keeps the FoodHub demo data "alive".

Every INTERVAL seconds (default 5) it does ONE change to data/foodhub_order.csv:
  - usually adds 1 new order, sometimes updates 1 old order (a customer finally leaves the
    rating that was 'Not given'), saved in the same file
  then runs the EDA and WAITS until it finishes before the next change, so
  processed/foodhub_order_output.csv and processed/foodhub.db are always rebuilt from a
  complete file.

Lives in setup/ (or the project root). Run from the project root (the folder that contains data/ and processed/):

    python setup/live_dashboard.py
    python setup/live_dashboard.py --eda-url http://127.0.0.1:8001/run-eda --interval 5

Stop with Ctrl+C. Runs until stopped.

FastAPI can also drive it: see LiveFeed (start / stop / status) and api/dashboard_api.py.
"""
from __future__ import annotations

import argparse
import os
import random
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional

import pandas as pd
import requests

_HERE = Path(__file__).resolve().parent
ROOT = _HERE.parent if _HERE.name.lower() in ("setup", "api", "agents") else _HERE   # project root either way
DEFAULT_CSV = ROOT / "data" / "foodhub_order.csv"
DEFAULT_EDA_URL = os.getenv("EDA_URL", "http://127.0.0.1:8000/run-eda")

REQUIRED = ["order_id", "customer_id", "restaurant_name", "cuisine_type", "cost_of_the_order",
            "day_of_the_week", "rating", "food_preparation_time", "delivery_time"]


def log(msg: str) -> None:
    print(f"[{datetime.now():%H:%M:%S}] {msg}", flush=True)


# ---------------------------------------------------------------------------
# CSV helpers (everything is read as text so existing values are never reformatted)
# ---------------------------------------------------------------------------
def load_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, dtype=str, keep_default_na=False)


def save_atomic(df: pd.DataFrame, path: Path) -> None:
    """Write to a temp file, then swap it in, so the EDA job never reads a half-written CSV."""
    tmp = path.with_name(path.name + ".tmp")
    df.to_csv(tmp, index=False)
    last = None
    for _ in range(20):                      # Windows: file may be briefly locked by Excel / the EDA job
        try:
            os.replace(tmp, path)
            return
        except PermissionError as e:
            last = e
            time.sleep(0.25)
    raise last


# ---------------------------------------------------------------------------
# Profile of the existing data -> new rows look like the real ones
# ---------------------------------------------------------------------------
class Profile:
    def __init__(self, df: pd.DataFrame):
        num = lambda c: pd.to_numeric(df[c], errors="coerce")
        rating_num = num("rating")
        self.no_rating = (df.loc[rating_num.isna(), "rating"].mode().iat[0]
                          if rating_num.isna().any() else "Not given")
        self.p_missing = float(rating_num.isna().mean())
        self.real_ratings = df.loc[rating_num.notna(), "rating"].tolist() or ["5", "4", "3"]

        counts = df["restaurant_name"].value_counts()
        self.rest_names = counts.index.tolist()
        self.rest_weights = counts.values.tolist()
        self.rest_cuisine = df.groupby("restaurant_name")["cuisine_type"].agg(lambda s: s.mode().iat[0]).to_dict()

        cost = num("cost_of_the_order")
        self.cost_min, self.cost_max = float(cost.min()), float(cost.max())
        self.cost_all = cost.dropna().tolist()
        self.cost_by_cuisine = {c: g.dropna().tolist() for c, g in cost.groupby(df["cuisine_type"])}

        days = df["day_of_the_week"].value_counts()
        self.days, self.day_weights = days.index.tolist(), days.values.tolist()
        prep, deliv = num("food_preparation_time"), num("delivery_time")
        self.prep_rng = (int(prep.min()), int(prep.max()))
        self.deliv_rng = (int(deliv.min()), int(deliv.max()))
        self.prep_by_day = {d: g.dropna().astype(int).tolist() for d, g in prep.groupby(df["day_of_the_week"])}
        self.deliv_by_day = {d: g.dropna().astype(int).tolist() for d, g in deliv.groupby(df["day_of_the_week"])}

        cid = num("customer_id").dropna().astype(int)
        self.cust_min, self.cust_max = int(cid.min()), int(cid.max())
        self.customer_pool = cid.astype(str).tolist()          # one entry per order = natural frequency
        self.customer_set = set(self.customer_pool)

    def new_order(self, rng: random.Random, order_id: int, columns: list) -> dict:
        rest = rng.choices(self.rest_names, weights=self.rest_weights)[0]
        cuisine = self.rest_cuisine[rest]
        day = rng.choices(self.days, weights=self.day_weights)[0]

        base = rng.choice(self.cost_by_cuisine.get(cuisine) or self.cost_all)
        cost = min(max(base * rng.uniform(0.9, 1.1), self.cost_min), self.cost_max)

        prep = rng.choice(self.prep_by_day.get(day) or [self.prep_rng[0]]) + rng.randint(-1, 1)
        deliv = rng.choice(self.deliv_by_day.get(day) or [self.deliv_rng[0]]) + rng.randint(-1, 1)
        prep = min(max(prep, self.prep_rng[0]), self.prep_rng[1])
        deliv = min(max(deliv, self.deliv_rng[0]), self.deliv_rng[1])

        rating = self.no_rating if rng.random() < self.p_missing else rng.choice(self.real_ratings)

        if rng.random() < 0.7:                                   # returning customer
            customer = rng.choice(self.customer_pool)
        else:                                                    # brand-new customer
            while True:
                customer = str(rng.randint(self.cust_min, self.cust_max))
                if customer not in self.customer_set:
                    break
        self.customer_pool.append(customer)
        self.customer_set.add(customer)

        row = {c: "" for c in columns}
        row.update({"order_id": str(order_id), "customer_id": customer, "restaurant_name": rest,
                    "cuisine_type": cuisine, "cost_of_the_order": f"{cost:.2f}", "day_of_the_week": day,
                    "rating": rating, "food_preparation_time": str(prep), "delivery_time": str(deliv)})
        return row


# ---------------------------------------------------------------------------
# One tick = ONE row added/updated, then the EDA runs to completion (same thread)
# ---------------------------------------------------------------------------
def http_eda(url: str, timeout: int = 300) -> Callable[[], str]:
    """EDA through a running API: POST url, block until it answers (= the EDA has finished)."""
    def run() -> str:
        r = requests.post(url, timeout=timeout)
        if not r.ok:                                         # show WHY the EDA API failed, not just the status
            raise RuntimeError(f"HTTP {r.status_code} from {url}: {r.text[:200]}")
        return "ok"
    return run


class LiveFeed:
    """Background feed: every `interval` seconds change ONE row in the CSV, then run the EDA and wait for it.

    eda_fn : function that rebuilds processed/ (e.g. FoodHubEDA(...).run). Called in the feed thread,
             so the next row is never touched while the EDA is still running.
    """

    def __init__(self, csv: Path = DEFAULT_CSV, eda_fn: Optional[Callable[[], object]] = None,
                 interval: float = 5.0, update_prob: float = 0.2, seed: Optional[int] = None):
        self.csv, self.eda_fn = Path(csv), eda_fn
        self.interval, self.update_prob = interval, update_prob
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        self._profile: Optional[Profile] = None
        self._rng = random.Random(seed)
        self.stats = {"ticks": 0, "rows_added": 0, "rows_updated": 0, "total_rows": None, "started_at": None,
                      "last_tick_at": None, "last_message": None, "last_error": None, "last_eda_seconds": None}

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    @property
    def stopping(self) -> bool:
        """True after stop() was requested while the current tick (EDA) is still finishing."""
        return self.running and self._stop.is_set()

    def _prepare(self) -> None:
        if not self.csv.exists():
            raise FileNotFoundError(f"CSV not found: {self.csv}")
        df = load_csv(self.csv)
        missing = [c for c in REQUIRED if c not in df.columns]
        if missing:
            raise ValueError(f"{self.csv.name} is missing columns {missing}; found {list(df.columns)}")
        if len(df) < 20:
            raise ValueError("Need at least 20 existing rows to learn realistic values from.")
        self._profile = Profile(df)
        self.stats["total_rows"] = len(df)

    def start(self) -> bool:
        """Start in a background thread. Returns False if it is already running."""
        with self._lock:
            if self.running:
                return False
            self._prepare()
            self._stop.clear()
            self.stats.update(ticks=0, rows_added=0, rows_updated=0, last_error=None,
                              started_at=datetime.now().isoformat(timespec="seconds"))
            self._thread = threading.Thread(target=self._loop, name="live-feed", daemon=True)
            self._thread.start()
            return True

    def stop(self, wait: float = 60.0) -> bool:
        with self._lock:
            if not self.running:
                return False
            self._stop.set()
            self._thread.join(timeout=wait)
            return True

    def status(self) -> dict:
        return {"running": self.running, "stopping": self.stopping, "interval_seconds": self.interval,
                "csv": str(self.csv), **self.stats}

    def tick(self) -> str:
        """Change exactly one row and save the CSV in place."""
        df = load_csv(self.csv)
        rng, prof = self._rng, self._profile
        unrated = df.index[df["rating"] == prof.no_rating][-200:]      # recent orders still without a rating

        if len(unrated) and rng.random() < self.update_prob:           # UPDATE one old row
            i = rng.choice(list(unrated))
            order_id, new_rating = df.at[i, "order_id"], rng.choice(prof.real_ratings)
            df.at[i, "rating"] = new_rating
            self.stats["rows_updated"] += 1
            msg = f"updated order {order_id}: rating -> {new_rating}"
        else:                                                           # ADD one new row
            order_id = int(pd.to_numeric(df["order_id"], errors="coerce").max()) + 1
            row = prof.new_order(rng, order_id, list(df.columns))
            df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
            self.stats["rows_added"] += 1
            msg = f"added order {order_id} ({row['restaurant_name']}, {row['cuisine_type']}, ${row['cost_of_the_order']})"

        save_atomic(df, self.csv)
        self.stats["total_rows"] = len(df)
        return f"{msg} | total {len(df):,} rows"

    def run_once(self) -> str:
        msg = self.tick()
        if self.eda_fn is not None:
            t0 = time.perf_counter()
            try:
                self.eda_fn()                                           # blocks until the EDA has finished
                self.stats["last_eda_seconds"] = round(time.perf_counter() - t0, 2)
                msg += f" | EDA done in {self.stats['last_eda_seconds']}s"
                self.stats["last_error"] = None
            except requests.exceptions.ConnectionError:
                msg += " | EDA API not reachable (is it running on the right port?)"
                self.stats["last_error"] = "EDA API not reachable"
            except Exception as e:
                msg += f" | EDA failed: {type(e).__name__}: {e}"
                self.stats["last_error"] = f"EDA failed: {e}"
        self.stats["ticks"] += 1
        self.stats["last_tick_at"] = datetime.now().isoformat(timespec="seconds")
        self.stats["last_message"] = msg
        return msg

    def _loop(self) -> None:
        log(f"live feed started (1 row every {self.interval:g}s, {self.stats['total_rows']:,} rows)")
        next_t = time.monotonic()
        while not self._stop.is_set():
            try:
                log(self.run_once())
            except Exception as e:                                      # keep running whatever happens
                self.stats["last_error"] = f"{type(e).__name__}: {e}"
                log(f"tick failed: {self.stats['last_error']}")
            next_t += self.interval
            wait = next_t - time.monotonic()
            if wait > 0:
                self._stop.wait(wait)
            else:                                                       # EDA took longer than the interval:
                next_t = time.monotonic()                               # go again right after it finished
        log("live feed stopped")

    def run_forever(self) -> None:
        """Blocking version for the command line."""
        self._prepare()
        self._stop.clear()
        self._loop()


def main() -> None:
    ap = argparse.ArgumentParser(description="Add/update ONE FoodHub order every few seconds, then run the EDA.")
    ap.add_argument("--csv", type=Path, default=DEFAULT_CSV, help="raw orders CSV (default data/foodhub_order.csv)")
    ap.add_argument("--eda-url", default=DEFAULT_EDA_URL, help="URL that runs the EDA (or set EDA_URL)")
    ap.add_argument("--interval", type=float, default=5.0, help="seconds between rows")
    ap.add_argument("--update-prob", type=float, default=0.2, help="chance a tick updates an old row instead of adding")
    ap.add_argument("--timeout", type=int, default=300, help="seconds to wait for the EDA")
    ap.add_argument("--no-eda", action="store_true", help="only change the CSV, do not run the EDA")
    ap.add_argument("--once", action="store_true", help="one tick then exit (testing)")
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args()

    feed = LiveFeed(args.csv, None if args.no_eda else http_eda(args.eda_url, args.timeout),
                    args.interval, args.update_prob, args.seed)
    try:
        if args.once:
            feed._prepare()
            log(feed.run_once())
        else:
            feed.run_forever()
    except (FileNotFoundError, ValueError) as e:
        sys.exit(str(e))
    except KeyboardInterrupt:
        log("stopped.")


if __name__ == "__main__":
    main()