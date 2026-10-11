"""
agents_hub.py  -  LLM-FREE version. Same public API as before (run_pipeline), no Ollama.

Pipeline (all deterministic, typically 5-30 ms per question):
  Question -> [0] Preprocessor -> [0] Parser/Classifier (slots: metrics, dimension, filters, direction, top-N)
           -> [1] Planner -> [2A] SQL compiler -> [2B] validate -> [2C] execute -> [2D] findings + insights
           -> [3] Aggregator (+ [3A] Trace) -> [4] Template report (insights + number-backed recommendations)

Why this is MORE accurate than the LLM version: SQL is built from whitelisted columns/aggregates, so it cannot
use a wrong column, add a filter nobody asked for, or invent a number. Every sentence in the report is
produced from a computed value.
"""

import json
import math
import os
import re
import sqlite3
import time
from pathlib import Path

import pandas as pd
import sqlparse
from rapidfuzz import fuzz, process

# ---------------------------------------------------------------------------
# STEP 0: CONFIG
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
DB_PATH = Path(os.getenv("FOODHUB_DB", ROOT / "processed" / "foodhub.db"))
MAX_ROWS = 50          # hard cap on rows returned
DEFAULT_LIMIT = 20     # rows when the user did not ask for "top N"
MIN_GROUP = 5          # minimum orders (or rated orders) for an AVERAGE ranking of customers/restaurants
SMALL_N = 10           # groups below this are flagged as small samples
USE_EMBEDDINGS = os.getenv("USE_EMBEDDINGS", "1") == "1"   # optional semantic fallback (not an LLM)

DAY = "day_of_the_week"


# ---------------------------------------------------------------------------
# STEP 0.5: TRACE AGENT (3A)
# ---------------------------------------------------------------------------
class Trace:
    def __init__(self):
        self.steps = []

    def log(self, step, **details):
        self.steps.append({"step": step, "time": time.strftime("%H:%M:%S"), **details})

    def to_json(self):
        return json.loads(json.dumps(self.steps, default=str))


# ---------------------------------------------------------------------------
# STEP 1: SCHEMA + CONTEXT (loaded ONCE, refreshed only if the DB file changes)
# ---------------------------------------------------------------------------
def norm(s) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9\s]", " ", str(s).lower())).strip()


def read_schema(db_path=DB_PATH) -> dict:
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    tables = [r[0] for r in conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'")]
    schema = {t: [(c[1], c[2]) for c in conn.execute(f"PRAGMA table_info('{t}')").fetchall()] for t in tables}
    conn.close()
    return schema


def pick_main_table(schema: dict) -> str:
    for table, cols in schema.items():
        if "cost_of_the_order" in [c[0] for c in cols]:
            return table
    return next(iter(schema))


def schema_to_text(schema: dict) -> str:
    return "\n".join(f"TABLE {t}({', '.join(f'{n} {ty}' for n, ty in cols)})" for t, cols in schema.items())


def metric_defs(cols: set) -> dict:
    """key -> (SQL expression, alias, human label). Only metrics whose columns exist are offered."""
    d = {
        "orders":       ("COUNT(*)", "orders", "orders"),
        "revenue":      ("ROUND(SUM(cost_of_the_order),2)", "revenue", "revenue"),
        "avg_cost":     ("ROUND(AVG(cost_of_the_order),2)", "avg_cost", "average order cost"),
        "avg_delivery": ("ROUND(AVG(delivery_time),1)", "avg_delivery_min", "average delivery time (min)"),
        "avg_prep":     ("ROUND(AVG(food_preparation_time),1)", "avg_prep_min", "average preparation time (min)"),
        "avg_total":    (f"ROUND(AVG({'total_time' if 'total_time' in cols else '(food_preparation_time + delivery_time)'}),1)",
                         "avg_total_min", "average total time (min)"),
        "customers":    ("COUNT(DISTINCT customer_id)", "customers", "unique customers"),
        "restaurants":  ("COUNT(DISTINCT restaurant_name)", "restaurants", "unique restaurants"),
    }
    if "commission" in cols:
        d["commission"] = ("ROUND(SUM(commission),2)", "commission", "total commission")
    if "total_time_sla" in cols:
        d["breach_pct"] = ("ROUND(100.0 * SUM(CASE WHEN total_time_sla = 'BREACHED' THEN 1 ELSE 0 END) / COUNT(*),1)",
                           "sla_breach_pct", "SLA breach rate (%)")
    if "rating_missing" in cols:     # imputed ratings (rating_missing = 1) are excluded from rating averages ONLY
        d["avg_rating"] = ("ROUND(AVG(CASE WHEN rating_missing = 0 THEN rating END),2)", "avg_rating", "average rating")
    elif "rating" in cols:
        d["avg_rating"] = ("ROUND(AVG(CASE WHEN rating <> 'Not given' THEN CAST(rating AS REAL) END),2)",
                           "avg_rating", "average rating")
    return d


_CTX = {}


def get_ctx(db_path=DB_PATH) -> dict:
    key = (str(db_path), os.path.getmtime(db_path))
    if _CTX.get("key") == key:
        return _CTX
    schema = read_schema(db_path)
    table = pick_main_table(schema)
    cols = {c[0] for c in schema[table]}
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)

    def distinct(col):
        return [r[0] for r in conn.execute(f"SELECT DISTINCT {col} FROM {table} WHERE {col} IS NOT NULL")] \
            if col in cols else []

    cuisines, restaurants, days = distinct("cuisine_type"), distinct("restaurant_name"), distinct(DAY)
    conn.close()

    day_map = {}
    for d in days:
        for k in ("weekday", "weekend"):
            if k in str(d).lower():
                day_map[k] = d
    rest_sorted = sorted({norm(r): r for r in restaurants if len(norm(r)) >= 4}.items(), key=lambda x: -len(x[0]))
    tokens = {w for n in list(map(norm, cuisines)) + [n for n, _ in rest_sorted] for w in n.split()}

    _CTX.clear()
    _CTX.update({
        "key": key, "db_path": db_path, "schema": schema, "schema_text": schema_to_text(schema),
        "table": table, "cols": cols, "metrics": metric_defs(cols),
        "cuisines": [(re.compile(rf"\b{re.escape(norm(c))}\b"), c) for c in cuisines],
        "restaurants": [(re.compile(rf"\b{re.escape(n)}\b"), r) for n, r in rest_sorted],
        "day_map": day_map, "entity_tokens": tokens,
    })
    return _CTX


# ---------------------------------------------------------------------------
# STEP 2: LAYER 0 - PREPROCESSOR (clean text + fix typos with RapidFuzz)
# ---------------------------------------------------------------------------
VOCAB = sorted({
    "customer", "customers", "restaurant", "restaurants", "cuisine", "cuisines", "revenue", "sales", "spending",
    "delivery", "preparation", "rating", "ratings", "satisfaction", "weekday", "weekend", "average", "orders",
    "commission", "popular", "fastest", "slowest", "highest", "lowest", "compare", "comparison", "trend",
    "overall", "summary", "overview", "expensive", "frequent", "repeat", "loyal", "segment", "breach", "total",
    "korean", "japanese", "mexican", "american", "indian", "italian", "mediterranean", "chinese", "thai",
    "french", "spanish", "vietnamese", "southern", "middle", "eastern",
})
STOP = {"which", "what", "who", "have", "with", "that", "from", "this", "their", "them", "than", "about", "does",
        "show", "give", "tell", "each", "every", "over", "under", "above", "below", "between", "versus", "into",
        "best", "worst", "most", "least", "top", "bottom", "much", "many", "time", "times", "cost", "price",
        "slow", "fast", "late", "kitchen", "cheap", "when", "where", "week", "days", "food", "order", "ordered"}


def preprocess_query(question: str) -> str:
    ctx = get_ctx()
    text = norm(question)
    protected = ctx["entity_tokens"] | STOP | set(VOCAB)
    fixed = []
    for w in text.split():
        if len(w) > 3 and not w.isdigit() and w not in protected:
            m = process.extractOne(w, VOCAB, scorer=fuzz.ratio, score_cutoff=85)   # "koren" -> "korean"
            w = m[0] if m else w
        fixed.append(w)
    text = " ".join(fixed)
    text = re.sub(r"\bcustomer (satisfaction|feedback|reviews?|ratings?)\b", r"\1", text)   # not a customer question
    return text


# ---------------------------------------------------------------------------
# STEP 3: LAYER 0 - PARSER / CLASSIFIER (question -> structured spec)
# ---------------------------------------------------------------------------
METRIC_PATTERNS = [(k, re.compile(p)) for k, p in [
    ("customers",    r"\b(?:unique|distinct|number of|how many|count of) customers\b"),
    ("restaurants",  r"\b(?:unique|distinct|number of|how many|count of) restaurants\b"),
    ("avg_total",    r"\b(?:total time|total delivery time|end to end(?: time)?|overall time)\b"),
    ("breach_pct",   r"\b(?:sla breach\w*|breach\w*|late orders?|late deliver\w*|delays?|delayed|late|sla)\b"),
    ("commission",   r"\bcommissions?\b"),
    ("avg_prep",     r"\b(?:food )?(?:prep|preparation|prepare)(?: time)?\b|\bkitchen\b"),
    ("avg_delivery", r"\bdeliver\w*(?: time)?\b|\bslow\w*\b|\bfast\w*\b|\bquick\w*\b|\bspeed\b"),
    ("avg_rating",   r"\b(?:ratings?|rated|satisfaction|reviews?|feedback|happy|happiest|unhappy)\b"),
    ("revenue",      r"\b(?:revenue|sales|earnings?|spend\w*|spent|spenders?|money|gmv|total (?:order )?cost)\b"),
    ("avg_cost",     r"\b(?:avg|average)(?: order)? (?:cost|value|price|spend)\b|\border (?:value|cost)\b|"
                     r"\bexpensive\b|\bpricey\b|\bcostly\b|\bcheap\w*\b|\bcost\b|\bprice\b"),
    ("orders",       r"\b(?:orders?|ordered|how many|number of|count|popular\w*|frequent\w*|repeat|loyal\w*|busiest)\b"),
    ("time_all",     r"\btime\b"),
]]

DIM_PATTERNS = [(k, re.compile(p)) for k, p in [          # multiword first so they mask the single words
    ("cuisine_tier",     r"\bcuisine tiers?\b"),
    ("customer_segment", r"\b(?:customer )?segments?\b"),
    ("price_band",       r"\bprice (?:bands?|tiers?)\b"),
    ("restaurant_name",  r"\brestaurants?\b|\boutlets?\b|\bvendors?\b"),
    ("cuisine_type",     r"\bcuisines?\b|\bfood types?\b"),
    ("customer_id",      r"\bcustomers?\b|\bbuyers?\b|\bspenders?\b|\busers?\b"),
]]

DIR_RULES = [(re.compile(p), d) for p, d in [
    (r"\b(?:fastest|quickest|shortest)\b|\bleast time\b", "asc"),
    (r"\b(?:slowest|longest|most delayed)\b", "desc"),
    (r"\b(?:lowest|least|fewest|smallest|minimum|cheapest|bottom)\b", "asc"),
    (r"\b(?:highest|most|largest|biggest|maximum|busiest|priciest)\b", "desc"),
    (r"\bbest\b", "best"), (r"\bworst\b", "worst"), (r"\btop\b", "desc"),
]]
LOWER_IS_BETTER = {"avg_delivery", "avg_prep", "avg_total", "breach_pct"}
ADDITIVE = {"orders", "revenue", "commission"}
OVERVIEW_METRICS = ["orders", "revenue", "avg_cost", "customers", "restaurants", "avg_delivery", "avg_prep",
                    "avg_rating", "breach_pct", "commission"]
DEFAULT_METRICS = {
    "customer_id": ["orders", "revenue"], "restaurant_name": ["orders", "revenue"],
    "cuisine_type": ["orders", "revenue", "avg_cost"], "customer_segment": ["orders", "revenue", "avg_cost"],
    "price_band": ["orders", "revenue", "avg_cost"], "cuisine_tier": ["orders", "revenue", "avg_cost"],
    DAY: ["orders", "revenue", "avg_cost", "avg_prep", "avg_delivery"],
}

# optional semantic fallback (sentence-transformers: an embedding model, NOT a generative LLM)
_SEM = {}
SEM_DESCRIPTIONS = {
    ("metric", "orders"): "number of orders, popularity, order volume",
    ("metric", "revenue"): "revenue, sales, total spending, earnings",
    ("metric", "avg_cost"): "average order cost, order value, price",
    ("metric", "avg_rating"): "customer ratings, satisfaction, reviews",
    ("metric", "avg_delivery"): "delivery time, delivery speed",
    ("metric", "avg_prep"): "food preparation time, kitchen speed",
    ("dim", "restaurant_name"): "restaurant performance",
    ("dim", "cuisine_type"): "cuisine type comparison",
    ("dim", "customer_id"): "customer behaviour, loyalty, top spenders",
}


def semantic_hint(text: str, kind: str, threshold: float = 0.40):
    """Closest metric/dim by cosine similarity; None if not confident or the library is missing."""
    if not USE_EMBEDDINGS:
        return None
    try:
        if "model" not in _SEM:
            from sentence_transformers import SentenceTransformer
            _SEM["model"] = SentenceTransformer("all-MiniLM-L6-v2")
            _SEM["keys"] = list(SEM_DESCRIPTIONS)
            _SEM["emb"] = _SEM["model"].encode(list(SEM_DESCRIPTIONS.values()), normalize_embeddings=True)
        q = _SEM["model"].encode(text, normalize_embeddings=True)
        sims = _SEM["emb"] @ q
        best = max((i for i, k in enumerate(_SEM["keys"]) if k[0] == kind), key=lambda i: sims[i], default=None)
        return _SEM["keys"][best][1] if best is not None and sims[best] >= threshold else None
    except Exception:
        return None


def _mask(rx, text):
    return rx.sub(lambda m: " " * len(m.group()), text)


def parse_question(q: str, ctx: dict) -> dict:
    text, notes = q, []

    # ---- entity values (restaurant names, cuisines) -> filters; masked so they never trigger keywords
    rest_hits = []
    for rx, name in ctx["restaurants"]:
        if rx.search(text):
            rest_hits.append(name)
            text = _mask(rx, text)
    cuis_hits = []
    for rx, name in ctx["cuisines"]:
        if rx.search(text):
            cuis_hits.append(name)
            text = _mask(rx, text)

    # ---- dimensions (before metrics so "spenders"/"customers" still count)
    dims, dtext = [], text
    for key, rx in DIM_PATTERNS:
        if rx.search(dtext):
            if key == "cuisine_type" and "cuisine_tier" in dims:
                continue
            dims.append(key)
            dtext = _mask(rx, dtext)

    # ---- metrics (ordered by first position in the question)
    hits = []
    mtext = re.sub(r"\b(?:cuisine tiers?|customer segments?|segments?|price (?:bands?|tiers?))\b", " ", text)
    for key, rx in METRIC_PATTERNS:
        m = rx.search(mtext)
        if m:
            hits.append((m.start(), key))
            mtext = _mask(rx, mtext)
    metrics = [k for _, k in sorted(hits)]
    if "time_all" in metrics:
        metrics.remove("time_all")
        if not {"avg_prep", "avg_delivery", "avg_total"} & set(metrics):
            metrics += ["avg_prep", "avg_delivery", "avg_total"]
    if "customers" in metrics and "customer_id" in dims:
        dims.remove("customer_id")                                       # "how many customers" is a count, not a ranking
    if "restaurants" in metrics and "restaurant_name" in dims:
        dims.remove("restaurant_name")

    # ---- day logic: filter (one day) vs split (compare / trend)
    has_wd, has_we = "weekday" in text, "weekend" in text
    explicit_split = bool(re.search(r"\b(trend\w*|over time|by day|day of the week|each day|per day)\b", text))
    generic_compare = bool(re.search(r"\b(compar\w*|versus|vs|difference|between)\b", text))
    entity_compare = len(cuis_hits) >= 2 or len(rest_hits) >= 2
    split_day = (has_wd and has_we) or explicit_split or (generic_compare and not entity_compare)
    filters = []
    if (has_wd ^ has_we) and not split_day:
        filters.append(f"{DAY} = '{ctx['day_map'].get('weekday' if has_wd else 'weekend', '')}'")
    if "weekday" in text and "weekend" in text and not generic_compare and not explicit_split:
        split_day = True

    # ---- entity filters
    def sql_in(col, vals):
        return f"{col} IN ({', '.join(repr(v) for v in vals)})"        # values come from the DB whitelist
    if rest_hits:
        filters.append(sql_in("restaurant_name", rest_hits))
    if cuis_hits:
        filters.append(sql_in("cuisine_type", cuis_hits))

    # ---- direction, top-N, singular
    asc, judged = None, None
    for rx, d in DIR_RULES:
        if rx.search(text):
            if d in ("best", "worst"):
                judged = d
            else:
                asc = (d == "asc")
            if asc is not None:
                break
    m = re.search(r"\b(?:top|best|worst|first|bottom|lowest|highest|slowest|fastest|least|most|cheapest)\s+(\d+)\b", text) \
        or re.search(r"\b(\d+)\s+(?:best|worst|top|restaurants?|customers?|cuisines?|segments?)\b", text)
    top_n = min(int(m.group(1)), MAX_ROWS) if m else None
    superlative = asc is not None or judged is not None
    plural = bool(re.search(r"\b(restaurants|customers|cuisines|segments)\b", text))
    singular = bool(re.match(r"(which|what|who)\b", text)) and superlative and not plural and top_n is None
    repeat = bool(re.search(r"\brepeat\w*\b", text))
    overview = bool(re.search(r"\b(overall|summary|overview|big picture|whole business)\b", text))

    # ---- semantic fallback when rules found nothing
    if not metrics and not dims and not filters and not overview:
        sm, sd = semantic_hint(q, "metric"), semantic_hint(q, "dim")
        metrics, dims = ([sm] if sm else []), ([sd] if sd else [])
        if sm or sd:
            notes.append("matched by semantic similarity (no exact keywords found)")

    # ---- assemble dimension list with sensible defaults
    avail = ctx["metrics"]
    metrics = [m for m in dict.fromkeys(metrics) if m in avail]
    if not dims:
        if split_day:
            dims = [DAY]
        elif rest_hits or cuis_hits or filters:
            dims = []                                                    # overall figures for the filtered slice
        elif metrics == ["avg_rating"]:
            dims = ["rating_dist"]
        elif metrics and not re.search(r"\b(total|overall|in all|altogether|how many|average)\b", text):
            dims = [DAY]
        elif metrics:
            dims = []
        elif len(cuis_hits) >= 2:
            dims = ["cuisine_type"]
    if len(cuis_hits) >= 2 and not dims:
        dims = ["cuisine_type"]
    if len(rest_hits) >= 2 and not dims:
        dims = ["restaurant_name"]
    if not dims and not metrics and not filters:
        overview = True                                                  # nothing specific -> big picture
        notes.append("could not match the question to a metric, dimension or filter; showing the overall summary")

    # ---- direction resolution
    primary = (metrics or DEFAULT_METRICS.get(dims[0] if dims else DAY, ["orders"]))[0]
    if asc is None:
        if judged == "best":
            asc = primary in LOWER_IS_BETTER
        elif judged == "worst":
            asc = primary not in LOWER_IS_BETTER
        else:
            asc = False

    if re.search(r"\b(over|above|more than|under|below|less than|greater than)\s+\d+", text):
        notes.append("numeric thresholds in the question (e.g. 'over 30') are not supported and were ignored")

    return {"dims": dims, "metrics": metrics, "filters": filters, "split_day": split_day, "asc": asc,
            "top_n": top_n, "singular": singular, "repeat": repeat, "overview": overview, "notes": notes,
            "entities": {"restaurants": rest_hits, "cuisines": cuis_hits}}


def classify_query(spec: dict) -> list:
    """Business domains touched by the question (kept for trace / API compatibility)."""
    dom = {"restaurant_name": "restaurant", "cuisine_type": "cuisine", "customer_id": "customer",
           "customer_segment": "customer", "price_band": "revenue", "cuisine_tier": "cuisine",
           DAY: "operations", "rating_dist": "satisfaction"}
    out = ["overview"] if spec["overview"] else []
    out += [dom[d] for d in spec["dims"] if d in dom]
    for m in spec["metrics"]:
        out.append({"avg_rating": "satisfaction", "avg_delivery": "operations", "avg_prep": "operations",
                    "avg_total": "operations", "breach_pct": "operations"}.get(m, "revenue"))
    return list(dict.fromkeys(out)) or ["overview"]


# ---------------------------------------------------------------------------
# STEP 4: LAYER 1 - PLANNER (one task per dimension, plus an overview task if asked)
# ---------------------------------------------------------------------------
def plan_tasks(spec: dict) -> list:
    tasks = []
    if spec["overview"]:
        tasks.append({"agent": "overview", "dim": None, "metrics": OVERVIEW_METRICS, **_common(spec)})
    for d in spec["dims"]:
        metrics = spec["metrics"] or DEFAULT_METRICS.get(d, ["orders", "revenue"])
        if d == "rating_dist":
            metrics = ["avg_rating"]
        tasks.append({"agent": d, "dim": d, "metrics": metrics, **_common(spec)})
    if not spec["dims"] and not spec["overview"]:
        tasks.append({"agent": "overall", "dim": None, "metrics": spec["metrics"] or OVERVIEW_METRICS, **_common(spec)})
    return tasks


def _common(spec):
    return {k: spec[k] for k in ("filters", "split_day", "asc", "top_n", "singular", "repeat", "notes")}


# ---------------------------------------------------------------------------
# STEP 5: LAYER 2A - SQL COMPILER (replaces the LLM: whitelisted pieces only)
# ---------------------------------------------------------------------------
def compile_sql(task: dict, ctx: dict) -> str:
    t, mdefs = ctx["table"], ctx["metrics"]
    keys = [k for k in task["metrics"] if k in mdefs]
    where = list(task["filters"])
    wsql = (" WHERE " + " AND ".join(where)) if where else ""
    dim = task["dim"]

    if dim == "rating_dist":                                             # real ratings only (no imputed values)
        cond = "rating_missing = 0" if "rating_missing" in ctx["cols"] else "rating <> 'Not given'"
        w = " WHERE " + " AND ".join(where + [cond])
        return f"SELECT rating, COUNT(*) AS orders FROM {t}{w} GROUP BY rating ORDER BY rating DESC"

    sel = [f"{mdefs[k][0]} AS {mdefs[k][1]}" for k in keys]
    if "avg_rating" in keys and "rating_missing" in ctx["cols"]:
        sel.append("SUM(CASE WHEN rating_missing = 0 THEN 1 ELSE 0 END) AS rated_orders")
    if "orders" not in keys and dim:
        sel.append("COUNT(*) AS n")
    if dim is None:
        return f"SELECT {', '.join(sel)} FROM {t}{wsql}"

    primary = keys[0]
    pexpr, palias = mdefs[primary][0], mdefs[primary][1]
    direction = "ASC" if task["asc"] else "DESC"
    is_avg = primary not in ADDITIVE and primary not in ("customers", "restaurants")
    cnt = ("SUM(CASE WHEN rating_missing = 0 THEN 1 ELSE 0 END)"
           if primary == "avg_rating" and "rating_missing" in ctx["cols"] else "COUNT(*)")
    having = []
    if is_avg and dim in ("customer_id", "restaurant_name") and not task.get("_no_having"):
        having.append(f"{cnt} >= {MIN_GROUP}")
    if task.get("repeat") and dim == "customer_id":
        having.append("COUNT(*) >= 2")
    hsql = (" HAVING " + " AND ".join(having)) if having else ""
    limit = min(task["top_n"] or (5 if task["singular"] else DEFAULT_LIMIT), MAX_ROWS)

    if dim == DAY:
        return f"SELECT {DAY}, {', '.join(sel)} FROM {t}{wsql} GROUP BY {DAY} ORDER BY {DAY}"

    if task["split_day"]:                                                # top-N entities, then split by day
        k = task["top_n"] or 10
        sub = (f"SELECT {dim} FROM {t}{wsql} GROUP BY {dim}{hsql} "
               f"ORDER BY {pexpr} IS NULL, {pexpr} {direction}, COUNT(*) DESC, {dim} LIMIT {k}")
        w2 = " WHERE " + " AND ".join(where + [f"{dim} IN ({sub})"])
        return (f"SELECT {dim}, {DAY}, {', '.join(sel)} FROM {t}{w2} "
                f"GROUP BY {dim}, {DAY} ORDER BY {dim}, {DAY}")

    return (f"SELECT {dim}, {', '.join(sel)} FROM {t}{wsql} GROUP BY {dim}{hsql} "
            f"ORDER BY {palias} IS NULL, {palias} {direction}, COUNT(*) DESC, {dim} LIMIT {limit}")


# ---------------------------------------------------------------------------
# STEP 6: LAYER 2B/2C - VALIDATOR + EXECUTOR (read-only)
# ---------------------------------------------------------------------------
FORBIDDEN = {"INSERT", "UPDATE", "DELETE", "DROP", "ALTER", "CREATE", "REPLACE", "ATTACH", "PRAGMA", "TRUNCATE"}


def validate_sql(sql: str) -> str:
    statements = [s for s in sqlparse.parse(sql) if str(s).strip()]
    if len(statements) != 1:
        raise ValueError("Only one SQL statement is allowed")
    stmt = statements[0]
    if stmt.get_type() != "SELECT":
        raise ValueError("Only SELECT statements are allowed")
    tokens = {t.value.upper() for t in stmt.flatten() if t.ttype and "Keyword" in str(t.ttype)}
    if tokens & FORBIDDEN:
        raise ValueError(f"Forbidden keyword: {tokens & FORBIDDEN}")
    clean = str(stmt).strip().rstrip(";").strip()
    if not re.search(r"\blimit\b", clean, flags=re.I):
        clean += f" LIMIT {MAX_ROWS}"
    return clean


def execute_sql(sql: str, db_path=DB_PATH) -> pd.DataFrame:
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        return pd.read_sql_query(sql, conn)
    finally:
        conn.close()


def totals(ctx: dict, filters: list) -> dict:
    """Whole-slice totals (same filters) so shares like 'top 5 = 31% of orders' are exact."""
    cols = ctx["cols"]
    parts = ["COUNT(*)", "ROUND(SUM(cost_of_the_order),2)",
             "ROUND(SUM(commission),2)" if "commission" in cols else "NULL",
             "SUM(rating_missing)" if "rating_missing" in cols else "NULL"]
    w = (" WHERE " + " AND ".join(filters)) if filters else ""
    row = execute_sql(validate_sql(f"SELECT {', '.join(parts)} FROM {ctx['table']}{w}"), ctx["db_path"]).iloc[0].tolist()
    return {"orders": row[0], "revenue": row[1], "commission": row[2], "unrated": row[3]}


# ---------------------------------------------------------------------------
# STEP 7: LAYER 2D - FINDINGS (DataFrame -> structured finding with insights + recommendations)
# ---------------------------------------------------------------------------
def fmtv(v) -> str:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "n/a"
    if isinstance(v, float) and not v.is_integer():
        return f"{v:,.2f}"
    return f"{int(v):,}"


def make_highlights(df: pd.DataFrame) -> list:
    """Deterministic highest/lowest per numeric column."""
    labels = [c for c in df.columns if not pd.api.types.is_numeric_dtype(df[c]) or c.endswith("_id")]
    nums = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c]) and not c.endswith("_id")
            and c not in ("n", "rated_orders")]
    if len(df) < 3 or not labels or not nums:
        return []
    name = lambda i: " / ".join(str(df.loc[i, l]) for l in labels)
    out = []
    for c in nums:
        col = df[c].dropna()
        if not col.empty:
            out.append(f"(only among these {len(df)} rows) {c}: highest = {name(col.idxmax())} ({col.max()}), "
                       f"lowest = {name(col.idxmin())} ({col.min()})")
    return out


DIM_LABEL = {"restaurant_name": "restaurant", "cuisine_type": "cuisine", "customer_id": "customer",
             "customer_segment": "customer segment", "price_band": "price band", "cuisine_tier": "cuisine tier"}


def _gap(a, b):
    return None if not b else (a - b) / b * 100


def analyze(task: dict, df: pd.DataFrame, ctx: dict, tot: dict):
    """Returns (insights, recommendations). Every number comes from the data or a Python computation on it."""
    ins, recs = [], []
    md, dim, keys = ctx["metrics"], task["dim"], [k for k in task["metrics"] if k in ctx["metrics"]]
    al = lambda k: md[k][1]
    lb = lambda k: md[k][2]
    scope = (" (filtered: " + "; ".join(task["filters"]) + ")") if task["filters"] else ""
    if df.empty:
        return [f"No rows matched this question{scope}."], []

    # ---- overall single row (overview / totals)
    if dim is None:
        r = df.to_dict("records")[0]
        ins.append("Overall" + scope + ": " + ", ".join(f"{lb(k)} = {fmtv(r[al(k)])}" for k in keys if al(k) in df))
        if "avg_rating" in keys and tot.get("unrated") and tot.get("orders"):
            pct = tot["unrated"] / tot["orders"] * 100
            ins.append(f"{fmtv(tot['unrated'])} of {fmtv(tot['orders'])} orders ({pct:.0f}%) have no real rating; "
                       "rating averages use rated orders only.")
            if pct >= 20:
                recs.append(f"Improve rating collection: {pct:.0f}% of orders ({fmtv(tot['unrated'])} of "
                            f"{fmtv(tot['orders'])}) are unrated, so rating insights rest on a partial sample.")
        return ins, recs

    # ---- rating distribution
    if dim == "rating_dist":
        total = df["orders"].sum()
        avg = (df["rating"] * df["orders"]).sum() / total if total else float("nan")
        mode = df.to_dict("records")[int(df["orders"].values.argmax())]
        ins.append(f"{fmtv(total)} rated orders{scope}: average rating {avg:.2f}; most common rating "
                   f"{fmtv(mode['rating'])} ({fmtv(mode['orders'])} orders).")
        ins.append("Distribution: " + ", ".join(f"{fmtv(r['rating'])} -> {fmtv(r['orders'])}" for r in df.to_dict("records")))
        if tot.get("unrated") and tot.get("orders"):
            pct = tot["unrated"] / tot["orders"] * 100
            ins.append(f"{pct:.0f}% of all orders ({fmtv(tot['unrated'])} of {fmtv(tot['orders'])}) carry no real rating.")
            if pct >= 20:
                recs.append(f"Improve rating collection: {pct:.0f}% of orders are unrated, which limits how far the "
                            "rating results can be trusted.")
        return ins, recs

    primary = keys[0]
    pa = al(primary)

    # ---- weekday vs weekend (dimension = day)
    if dim == DAY:
        if len(df) >= 2:
            for k in keys:
                srt = df.sort_values(al(k), ascending=False).to_dict("records")
                a, b = srt[0], srt[-1]
                g = _gap(a[al(k)], b[al(k)])
                if g is None or a[al(k)] == b[al(k)]:
                    ins.append(f"{lb(k)} is identical on both day types ({fmtv(a[al(k)])}).")
                    continue
                ins.append(f"{lb(k)}: {a[DAY]} {fmtv(a[al(k)])} vs {b[DAY]} {fmtv(b[al(k)])} ({g:.0f}% higher on {a[DAY]}).")
                if k in LOWER_IS_BETTER and g >= 10:
                    recs.append(f"Investigate {a[DAY]} performance: {lb(k)} is {fmtv(a[al(k)])} vs {fmtv(b[al(k)])} on "
                                f"{b[DAY]} ({g:.0f}% worse).")
                elif k in ("orders", "revenue") and g >= 25:
                    recs.append(f"{b[DAY]} trails {a[DAY]} on {lb(k)} ({fmtv(b[al(k)])} vs {fmtv(a[al(k)])}); "
                                f"review what limits {b[DAY]} demand.")
        else:
            ins.append("Only one day type present: " + ", ".join(f"{lb(k)} = {fmtv(df.to_dict('records')[0][al(k)])}" for k in keys))
        return ins, recs

    # ---- entity x day split
    if task["split_day"] and DAY in df.columns:
        piv = df.pivot_table(index=dim, columns=DAY, values=pa, aggfunc="first")
        label = DIM_LABEL.get(dim, dim)
        ins.append(f"{lb(primary).capitalize()} by {label} and day type for {piv.shape[0]} {label}s{scope}.")
        if piv.shape[1] >= 2:
            c1, c2 = piv.columns[:2]
            full = piv.dropna()
            one_sided = piv.shape[0] - len(full)
            if one_sided:
                ins.append(f"{one_sided} {label}(s) appear on only one day type, so no comparison is possible for them.")
            if not full.empty:
                gap = full.apply(lambda r: _gap(r[c1], r[c2]), axis=1).dropna()
                for name in gap.abs().sort_values(ascending=False).index[:3]:
                    ins.append(f"{name}: {c1} {fmtv(full.loc[name, c1])} vs {c2} {fmtv(full.loc[name, c2])} "
                               f"({gap[name]:+.0f}% {c1} vs {c2}).")
                ahead = int((full[c1] > full[c2]).sum())
                ins.append(f"{c1} is higher for {ahead} of {len(full)} {label}s; {c2} for {int((full[c2] > full[c1]).sum())}.")
        return ins, recs

    # ---- ranked list (default)
    label, plural = DIM_LABEL.get(dim, dim), DIM_LABEL.get(dim, dim) + "s"
    valid = df[df[pa].notna()]
    if valid.empty:
        return [f"All {plural} have no value for {lb(primary)}{scope}."], []
    order = "lowest" if task["asc"] else "highest"
    recs_ = valid.to_dict("records")
    cntc = "rated_orders" if primary == "avg_rating" and "rated_orders" in valid else ("n" if "n" in valid else None)
    show = (lambda r: f"{fmtv(r[pa])}, n={fmtv(r[cntc])}") if (cntc and primary not in ADDITIVE) else (lambda r: fmtv(r[pa]))
    if task["singular"]:
        r = recs_[0]
        ins.append(f"ANSWER: the {label} with the {order} {lb(primary)} is {r[dim]} ({fmtv(r[pa])}){scope}.")
    ins.append(f"{plural.capitalize()} ranked by {lb(primary)} ({order} first, {len(valid)} shown): "
               + ", ".join(f"{r[dim]} ({show(r)})" for r in recs_[:5]) + ("..." if len(valid) > 5 else "."))
    topv = valid[pa].iloc[0]
    tied = valid[valid[pa] == topv][dim].astype(str).tolist()
    if len(tied) > 1:
        ins.append(f"{len(tied)} {plural} tie at {fmtv(topv)}: {', '.join(tied[:6])}{'...' if len(tied) > 6 else ''}.")
    others = [k for k in keys[1:] if al(k) in valid]
    if others:
        r = recs_[0]
        ins.append(f"The leader, {r[dim]}, also has " + ", ".join(f"{lb(k)} {fmtv(r[al(k)])}" for k in others) + ".")
    if len(valid) >= 3:
        srt = valid.sort_values(pa).to_dict("records")
        lo, hi = srt[0], srt[-1]
        ins.append(f"Among the {len(valid)} {plural} shown, {lb(primary)} runs from {fmtv(lo[pa])} ({lo[dim]}) "
                   f"to {fmtv(hi[pa])} ({hi[dim]}).")

    if primary in ADDITIVE and tot.get(primary):
        share = valid[pa].sum() / tot[primary] * 100
        ins.append(f"These {len(valid)} {plural} account for {share:.0f}% of all {lb(primary)} "
                   f"({fmtv(valid[pa].sum())} of {fmtv(tot[primary])}){scope}.")
        if share >= 99.9:
            ins.pop()                                                    # trivial: the rows shown are everything
        elif share >= 40 and len(valid) <= 10:
            recs.append(f"Protect the key {plural}: the {len(valid)} listed generate {share:.0f}% of {lb(primary)} "
                        f"({fmtv(valid[pa].sum())} of {fmtv(tot[primary])}), so losing one would be material.")

    cnt = cntc or "orders"
    if primary not in ADDITIVE and cnt in valid:
        small = valid[valid[cnt] < SMALL_N]
        if len(small):
            ins.append(f"Caution: {len(small)} of {len(valid)} {plural} have fewer than {SMALL_N} "
                       f"{'rated ' if cnt == 'rated_orders' else ''}orders, so their averages are unreliable.")
        solid = valid[valid[cnt] >= SMALL_N]
        if len(solid) >= 2 and primary in LOWER_IS_BETTER:
            srt = solid.sort_values(pa).to_dict("records")
            slow, fast = srt[-1], srt[0]
            g = _gap(slow[pa], fast[pa])
            if g is not None and g >= 10:
                recs.append(f"Review {label} {slow[dim]} ({fmtv(slow[pa])}) against {fast[dim]} ({fmtv(fast[pa])}): "
                            f"{lb(primary)} differs by {g:.0f}% among {plural} with at least {SMALL_N} orders.")
        if len(solid) >= 2 and primary == "avg_rating":
            low = solid.sort_values(pa).to_dict("records")[0]
            recs.append(f"Look into low-rated {label} {low[dim]}: average rating {fmtv(low[pa])} over "
                        f"{fmtv(low[cnt])} rated orders.")
    if task["repeat"] and dim == "customer_id":
        ins.append("Repeat customers = customers with at least 2 orders.")
    return ins, recs


def make_finding(task: dict, sql: str, df: pd.DataFrame, ctx: dict) -> dict:
    df = df.round(2)
    tot = totals(ctx, task["filters"])
    insights, recs = analyze(task, df, ctx, tot)
    if df.empty:
        summary = f"{task['agent']}: query returned no rows."
    else:
        rows = [", ".join(f"{c}={r[c]}" for c in df.columns) for r in df.head(6).to_dict("records")]
        summary = f"{task['agent']}: {len(df)} rows. " + " | ".join(rows) + (" ..." if len(df) > 6 else "")
    return {"agent": task["agent"], "sql": sql, "sql_source": "compiled", "summary": summary,
            "highlights": make_highlights(df), "insights": insights, "recommendations": recs,
            "notes": task["notes"], "data": df.head(20).to_dict(orient="records")}


def run_domain_agent(task: dict, ctx: dict, trace: Trace) -> dict:
    """2A compile -> 2B validate -> 2C execute -> 2D findings. If HAVING removes every group, retry once without it."""
    sql = compile_sql(task, ctx)
    trace.log("2A sql_compiled", agent=task["agent"], sql=sql)
    safe = validate_sql(sql)
    df = execute_sql(safe, ctx["db_path"])
    if (df.empty or (len(df) < 3 and not task["top_n"])) and " HAVING " in safe and task["dim"] \
            and not task.get("repeat"):
        task = {**task, "_no_having": True}
        sql = compile_sql(task, ctx)
        safe = validate_sql(sql)
        df = execute_sql(safe, ctx["db_path"])
        task["notes"] = task["notes"] + ["minimum-group-size filter removed because no group met it"]
        trace.log("2B relaxed_having", agent=task["agent"], sql=safe)
    trace.log("2C executed", agent=task["agent"], rows=len(df))
    return make_finding(task, safe, df, ctx)


# ---------------------------------------------------------------------------
# STEP 8: LAYER 3 - AGGREGATOR
# ---------------------------------------------------------------------------
def aggregate(findings: list) -> dict:
    seen, unique = {}, []
    for f in findings:
        key = str([list(row.values()) for row in f["data"]])
        if key in seen:
            seen[key]["agent"] += f" + {f['agent']}"
        else:
            seen[key] = f
            unique.append(f)
    return {"count": len(unique), "summaries": [f["summary"] for f in unique], "findings": unique}


# ---------------------------------------------------------------------------
# STEP 9: LAYER 4 - TEMPLATE REPORT (no LLM)
# ---------------------------------------------------------------------------
def executive_report(question: str, unified: dict) -> str:
    insights, recs, notes = [], [], []
    for f in unified["findings"]:
        insights += f["insights"]
        recs += f["recommendations"]
        notes += f["notes"]
    answers = [i for i in insights if i.startswith("ANSWER:")]
    if answers and len(unified["findings"]) == 1:                        # simple factual lookup: 1-2 sentences
        extra = [i for i in insights if not i.startswith("ANSWER:")][:1]
        return " ".join(answers + extra)
    out = ["KEY INSIGHTS"] + [f"- {i}" for i in dict.fromkeys(insights)]
    if recs:
        out += ["", "RECOMMENDATIONS"] + [f"{n}. {r}" for n, r in enumerate(dict.fromkeys(recs), 1) if n <= 3]
    if notes:
        out += ["", "NOTES"] + [f"- {n}" for n in dict.fromkeys(notes)]
    return "\n".join(out)


# ---------------------------------------------------------------------------
# STEP 10: ORCHESTRATOR
# ---------------------------------------------------------------------------
def json_safe(obj):
    if isinstance(obj, dict):
        return {k: json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [json_safe(v) for v in obj]
    if isinstance(obj, float):
        return None if (math.isnan(obj) or math.isinf(obj)) else float(obj)
    if hasattr(obj, "item"):
        return json_safe(obj.item())
    return obj


def run_pipeline(question: str) -> dict:
    trace = Trace()
    ctx = get_ctx()
    trace.log("1 schema", main_table=ctx["table"])

    clean_q = preprocess_query(question)                                  # Layer 0
    trace.log("0 preprocess", original=question, clean=clean_q)

    spec = parse_question(clean_q, ctx)                                   # Layer 0
    agents = classify_query(spec)
    trace.log("0 parse", spec=spec, agents=agents)

    plan = plan_tasks(spec)                                               # Layer 1
    trace.log("1 plan", tasks=[t["agent"] for t in plan])

    findings = [run_domain_agent(t, ctx, trace) for t in plan]            # Layer 2 (ms-level, no threads needed)

    unified = aggregate(findings)                                         # Layer 3
    trace.log("3 aggregate", findings=unified["count"])

    report = executive_report(question, unified)                          # Layer 4
    trace.log("4 report", chars=len(report))

    return json_safe({"question": question, "clean_question": clean_q, "agents": agents,
                      "report": report, "findings": unified["findings"], "trace": trace.to_json()})


if __name__ == "__main__":
    import sys
    q = " ".join(sys.argv[1:]) or "overall summary"
    t0 = time.perf_counter()
    res = run_pipeline(q)
    print(res["report"])
    print(f"\n[{(time.perf_counter() - t0) * 1000:.1f} ms]")