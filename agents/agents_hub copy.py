"""
agents_hub.py  -  all the AGENT LOGIC lives here (Layers 0 -> 4).
agents_api.py  -  only exposes this logic (CLI / REST).

Pipeline:
  Question -> [0] Preprocessor -> [0] Classifier -> [1] Planner + Schema
           -> [2] Domain agents -> [2A] SQL gen -> [2B] validate -> [2C] execute -> [2D] findings
           -> [3] Aggregator (+ [3A] Trace) -> [4] Executive report
"""

import json
import math
import os
import re
import sqlite3
import time
from pathlib import Path

import pandas as pd
import requests
import sqlparse
from rapidfuzz import process

from concurrent.futures import ThreadPoolExecutor
import threading

# ---------------------------------------------------------------------------
# STEP 0: CONFIG
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]                      # project root (FOODHUB_...)
DB_PATH = Path(os.getenv("FOODHUB_DB", ROOT / "processed" / "foodhub.db"))
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434/api/chat")  # local Qwen3 via Ollama
# Two models: a code model for SQL, a general model for the written report (override via env vars)
SQL_MODEL = os.getenv("SQL_MODEL", "qwen2.5-coder:7b")
REPORT_MODEL = os.getenv("REPORT_MODEL", "qwen3-vl:8b-instruct")
MAX_ROWS = 50                                                   # safety cap on query output


# ---------------------------------------------------------------------------
# STEP 0.5: TRACE AGENT (3A)  - records every step so you can see the lineage
# ---------------------------------------------------------------------------
class Trace:
    def __init__(self):
        self.steps = []

    def log(self, step, **details):
        self.steps.append({"step": step, "time": time.strftime("%H:%M:%S"), **details})

    def to_json(self):
        return json.loads(json.dumps(self.steps, default=str))


# ---------------------------------------------------------------------------
# STEP 1: LAYER 0 - QUERY PREPROCESSOR  (clean text + fix typos with RapidFuzz)
# ---------------------------------------------------------------------------
# Domain keywords: used for typo-fixing AND for classification
DOMAIN_KEYWORDS = {
    "customer":     ["customer", "customers", "repeat", "loyal", "frequent", "buyer"],
    "restaurant":   ["restaurant", "restaurants", "outlet", "shop"],
    "cuisine":      ["cuisine", "cuisines", "food", "korean", "japanese", "mexican", "american",
                     "indian", "italian", "mediterranean", "chinese", "thai", "french",
                     "spanish", "vietnamese", "southern"],
    "revenue":      ["revenue", "sales", "cost", "spend", "spending", "money", "price", "earning"],
    "operations":   ["delivery", "preparation", "prep", "time", "sla", "late", "slow", "fast",
                     "weekend", "weekday", "delay"],
    "satisfaction": ["rating", "ratings", "satisfaction", "review", "feedback", "happy", "rated"],
}
VOCAB = sorted({w for words in DOMAIN_KEYWORDS.values() for w in words})


def preprocess_query(question: str) -> str:
    text = question.lower().strip()
    text = re.sub(r"[^a-z0-9\s]", " ", text)           # remove punctuation
    text = re.sub(r"\s+", " ", text)                    # collapse spaces
    fixed = []
    for word in text.split():
        if len(word) > 3 and word not in VOCAB:         # try to fix typos like "koren" -> "korean"
            match = process.extractOne(word, VOCAB, score_cutoff=85)
            word = match[0] if match else word
        fixed.append(word)
    return " ".join(fixed)


# ---------------------------------------------------------------------------
# STEP 2: LAYER 0 - CLASSIFIER AGENT  (which business domains are needed?)
# ---------------------------------------------------------------------------
_embedder = None  # loaded lazily (sentence-transformers is optional, rules always work)

DOMAIN_DESCRIPTIONS = {
    "customer":     "customer behaviour, repeat orders, top spenders, loyalty",
    "restaurant":   "restaurant performance, best and worst restaurants",
    "cuisine":      "cuisine popularity, cuisine type comparison",
    "revenue":      "revenue, order cost, sales, spending",
    "operations":   "delivery time, food preparation time, SLA, weekday vs weekend speed",
    "satisfaction": "ratings, customer satisfaction, feedback",
}


def classify_query(question: str) -> list:
    # Rule-based pass: any keyword hit selects the domain
    words = set(question.split())
    selected = [d for d, kws in DOMAIN_KEYWORDS.items() if words & set(kws)]

    # Semantic pass (optional): add the closest domain by embedding similarity
    try:
        global _embedder
        from sentence_transformers import SentenceTransformer, util
        if _embedder is None:
            _embedder = SentenceTransformer("all-MiniLM-L6-v2")
        names = list(DOMAIN_DESCRIPTIONS)
        q_emb = _embedder.encode(question, convert_to_tensor=True)
        d_emb = _embedder.encode(list(DOMAIN_DESCRIPTIONS.values()), convert_to_tensor=True)
        scores = util.cos_sim(q_emb, d_emb)[0]
        best = names[int(scores.argmax())]
        if scores.max() > 0.30 and best not in selected:
            selected.append(best)
    except Exception:
        pass  # no sentence-transformers installed -> rules only

    selected = selected or ["revenue"]                      # sensible default
    if re.search(r"overall|summary|overview|big picture|whole business", question):
        selected = ["overview"] + selected                  # add real totals for broad questions
    return selected


# ---------------------------------------------------------------------------
# STEP 3: LAYER 1 - SCHEMA AGENT  (read tables/columns using SQLite PRAGMA)
# ---------------------------------------------------------------------------
def read_schema(db_path=DB_PATH) -> dict:
    conn = sqlite3.connect(db_path)
    tables = [r[0] for r in conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'")]
    schema = {}
    for t in tables:
        cols = conn.execute(f"PRAGMA table_info('{t}')").fetchall()   # (cid, name, type, ...)
        schema[t] = [(c[1], c[2]) for c in cols]
    conn.close()
    return schema


def pick_main_table(schema: dict) -> str:
    # The main table is the one that has the order cost column
    for table, cols in schema.items():
        if "cost_of_the_order" in [c[0] for c in cols]:
            return table
    return next(iter(schema))


def schema_to_text(schema: dict) -> str:
    return "\n".join(f"TABLE {t}({', '.join(f'{n} {ty}' for n, ty in cols)})"
                     for t, cols in schema.items())


# ---------------------------------------------------------------------------
# STEP 4: LAYER 1 - PLANNER AGENT  (one task per selected domain)
# ---------------------------------------------------------------------------
def extract_top_n(question: str, default=None):
    m = re.search(r"\b(?:top|best|worst|first)\s+(\d+)", question)      # "top 5" -> 5
    return min(int(m.group(1)), MAX_ROWS) if m else default


def is_trend(question: str) -> bool:
    # The data has NO date column, so "trend" means: compare Weekday vs Weekend
    return bool(re.search(r"trend|over time|compare|comparison|weekday|weekend|by day", question))


def plan_tasks(question: str, agents: list) -> list:
    n, trend = extract_top_n(question), is_trend(question)
    return [{"agent": a, "question": question, "top_n": n, "trend": trend} for a in agents]


# ---------------------------------------------------------------------------
# STEP 5: LAYER 2 - DOMAIN AGENTS (Customer, Restaurant, Cuisine, Revenue, Operations, Satisfaction)
# Each agent = a focus instruction for the SQL generator + a safe fallback SQL.
# ---------------------------------------------------------------------------
DOMAIN_AGENTS = {
    "overview": {"focus": "overall business totals", "fallback": ""},   # SQL built by overview_sql()
    "customer": {
        "focus": "customer behaviour: orders per customer_id, total spend, repeat customers",
        "fallback": """SELECT customer_id, COUNT(*) AS orders, ROUND(SUM(cost_of_the_order),2) AS total_spent
                       FROM {t} GROUP BY customer_id ORDER BY orders DESC, total_spent DESC LIMIT {n}""",
    },
    "restaurant": {
        "focus": "restaurant performance: orders and revenue per restaurant_name",
        "fallback": """SELECT restaurant_name, COUNT(*) AS orders, ROUND(SUM(cost_of_the_order),2) AS revenue
                       FROM {t} GROUP BY restaurant_name ORDER BY orders DESC LIMIT {n}""",
        # "trend" = top-N restaurants split by day_of_the_week (no dates exist in the data)
        "trend_fallback": """SELECT restaurant_name, day_of_the_week, COUNT(*) AS orders,
                                    ROUND(SUM(cost_of_the_order),2) AS revenue
                             FROM {t}
                             WHERE restaurant_name IN (SELECT restaurant_name FROM {t}
                                                       GROUP BY restaurant_name
                                                       ORDER BY COUNT(*) DESC LIMIT {n})
                             GROUP BY restaurant_name, day_of_the_week
                             ORDER BY restaurant_name, day_of_the_week""",
    },
    "cuisine": {
        "focus": "cuisine analysis: orders, revenue and average cost per cuisine_type",
        "fallback": """SELECT cuisine_type, COUNT(*) AS orders, ROUND(SUM(cost_of_the_order),2) AS revenue,
                              ROUND(AVG(cost_of_the_order),2) AS avg_cost
                       FROM {t} GROUP BY cuisine_type ORDER BY orders DESC LIMIT {n}""",
        "trend_fallback": """SELECT cuisine_type, day_of_the_week, COUNT(*) AS orders,
                                    ROUND(SUM(cost_of_the_order),2) AS revenue
                             FROM {t}
                             WHERE cuisine_type IN (SELECT cuisine_type FROM {t}
                                                    GROUP BY cuisine_type ORDER BY COUNT(*) DESC LIMIT {n})
                             GROUP BY cuisine_type, day_of_the_week
                             ORDER BY cuisine_type, day_of_the_week""",
    },
    "revenue": {
        "focus": "revenue: SUM and AVG of cost_of_the_order, split by day_of_the_week",
        "fallback": """SELECT day_of_the_week, COUNT(*) AS orders, ROUND(SUM(cost_of_the_order),2) AS revenue,
                              ROUND(AVG(cost_of_the_order),2) AS avg_order_value
                       FROM {t} GROUP BY day_of_the_week""",
    },
    "operations": {
        "focus": "operations/SLA: average food_preparation_time and delivery_time by day_of_the_week",
        "fallback": """SELECT day_of_the_week, ROUND(AVG(food_preparation_time),1) AS avg_prep_min,
                              ROUND(AVG(delivery_time),1) AS avg_delivery_min,
                              ROUND(AVG(food_preparation_time + delivery_time),1) AS avg_total_min
                       FROM {t} GROUP BY day_of_the_week""",
    },
    "satisfaction": {
        "focus": "satisfaction: distribution of rating (note: 'Not given' means no rating)",
        "fallback": """SELECT rating, COUNT(*) AS orders FROM {t} GROUP BY rating ORDER BY orders DESC""",
    },
}


def overview_sql(schema_text: str, table: str) -> str:
    rating = ("ROUND(AVG(CASE WHEN rating_missing = 0 THEN rating END),2)" if "rating_missing" in schema_text
              else "NULL")
    return (f"SELECT COUNT(*) AS total_orders, ROUND(SUM(cost_of_the_order),2) AS total_revenue, "
            f"ROUND(AVG(cost_of_the_order),2) AS avg_order_value, COUNT(DISTINCT customer_id) AS customers, "
            f"COUNT(DISTINCT restaurant_name) AS restaurants, ROUND(AVG(delivery_time),1) AS avg_delivery_min, "
            f"ROUND(AVG(food_preparation_time),1) AS avg_prep_min, {rating} AS avg_rating FROM {table}")


def get_fallback_sql(task: dict, table: str) -> str:
    agent = DOMAIN_AGENTS[task["agent"]]
    tpl = agent["trend_fallback"] if task.get("trend") and "trend_fallback" in agent else agent["fallback"]
    return tpl.format(t=table, n=task.get("top_n") or 10)


# ---------------------------------------------------------------------------
# STEP 6: LLM HELPER  (Qwen3 through Ollama; returns None if unavailable)
# ---------------------------------------------------------------------------
_local = threading.local()
_llm_cache = {}

def get_llm_error():
    return getattr(_local, "error", None)

def call_llm(system: str, user: str, model: str, num_predict: int = 400):
    key = (model, system, user)
    if key in _llm_cache:
        return _llm_cache[key]
    _local.error = None
    try:
        resp = requests.post(OLLAMA_URL, timeout=120, json={
            "model": model, "stream": False, "keep_alive": "30m",
            "options": {"temperature": 0, "num_predict": num_predict},
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
        })
        data = resp.json()
        if "message" not in data:
            _local.error = str(data.get("error", data))
            return None
        text = re.sub(r"<think>.*?</think>", "", data["message"]["content"], flags=re.S).strip()
        _llm_cache[key] = text
        return text
    except Exception as e:
        _local.error = f"{type(e).__name__}: {e}"
        return None


# ---------------------------------------------------------------------------
# STEP 7: LAYER 2A - SQL GENERATION AGENT
# ---------------------------------------------------------------------------
def build_data_notes(schema_text: str) -> str:
    notes = ["Use only columns that exist in the schema.",
             "Never GROUP BY a measure column (delivery_time, food_preparation_time, cost_of_the_order, rating).",
             "When ranking groups by an AVERAGE, include COUNT(*) AS n and HAVING COUNT(*) >= 5. "
             "Do NOT use HAVING for rankings by order count, revenue (SUM) or top-N requests.",
             "If the question asks for two metrics (e.g. rating AND delivery time), return BOTH metrics "
             "for ALL groups (LIMIT 20), sorted by the first metric asked."]
    if "rating_missing" in schema_text:
        notes.append("rating_missing = 1 marks imputed (fake) ratings. Use WHERE rating_missing = 0 ONLY when "
                     "averaging the rating column itself. NEVER use it for customer spend, order cost, "
                     "delivery_time, food_preparation_time or order counts: those must use ALL orders.")
        notes.append("Do not add filters the user did not ask for (e.g. delivery_time <= 30).")
    else:
        notes.append("rating may be text; 'Not given' means no rating: exclude it before averaging.")
    return "Data notes:\n- " + "\n- ".join(notes) + "\n"


def generate_sql(task: dict, schema_text: str, table: str, feedback: str = None, prev_sql: str = None) -> tuple:
    agent = DOMAIN_AGENTS[task["agent"]]
    system = ("You write SQLite queries. Return ONLY one SELECT statement, no markdown, no explanation. "
              f"Always use LIMIT {MAX_ROWS} or fewer.")
    user = (f"Schema:\n{schema_text}\n\n{build_data_notes(schema_text)}\nFocus: {agent['focus']}\n"
            f"User question: {task['question']}\n"
            f"{('Return at most ' + str(task['top_n']) + ' entities.') if task.get('top_n') else 'No count was asked: return ALL groups (LIMIT 20).'} "
            f"The data has NO date column; "
            f"for any 'trend' question group by day_of_the_week (Weekday/Weekend).\n"
            f"Write the SQL.")
    if task.get("trend") and "trend_fallback" in agent:
        example = agent["trend_fallback"].format(t=table, n=task.get("top_n") or 10)
        user += ("\nFollow this exact pattern (top N entities chosen in a subquery, "
                 "then split by day_of_the_week):\n" + example)
    if feedback:                                   # second attempt: tell the model what was wrong
        user += (f"\n\nYour previous SQL was REJECTED.\nPrevious SQL: {prev_sql}\nReason: {feedback}\n"
                 f"Write a corrected SQL that fixes this.")
    sql = call_llm(system, user, SQL_MODEL)
    if sql:
        sql = re.sub(r"```(?:sql)?|```", "", sql).strip()
        return sql, "llm"
    return get_fallback_sql(task, table), "fallback_template"        # works without an LLM


# ---------------------------------------------------------------------------
# STEP 8: LAYER 2B - SQL VALIDATOR  (sqlparse; only safe, single SELECT)
# ---------------------------------------------------------------------------
FORBIDDEN = {"INSERT", "UPDATE", "DELETE", "DROP", "ALTER", "CREATE", "REPLACE", "ATTACH", "PRAGMA", "TRUNCATE"}


def fix_rating_filter(sql: str) -> str:
    """LLMs wrongly add WHERE rating_missing = 0, which drops 736 unrated orders from cost/delivery metrics.
    Fix in code: remove the filter and apply it only inside AVG(rating)."""
    if not re.search(r"rating_missing\s*=\s*0", sql, flags=re.I):
        return sql
    n_rating = len(re.findall(r"\brating\b", sql, flags=re.I))
    avg_pat = r"AVG\(\s*rating\s*\)"
    if n_rating != len(re.findall(avg_pat, sql, flags=re.I)):
        return sql                                         # rating used some other way: leave it to the guard
    sql = re.sub(avg_pat, "AVG(CASE WHEN rating_missing = 0 THEN rating END)", sql, flags=re.I)
    sql = re.sub(r"WHERE\s+rating_missing\s*=\s*0\s+AND\s+", "WHERE ", sql, flags=re.I)   # WHERE rm=0 AND x
    sql = re.sub(r"\s+AND\s+rating_missing\s*=\s*0\b", "", sql, flags=re.I)               # WHERE x AND rm=0
    sql = re.sub(r"\s*WHERE\s+rating_missing\s*=\s*0\b", "", sql, flags=re.I)             # WHERE rm=0 (only)
    return sql


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
    if not re.search(r"\blimit\b", clean, flags=re.I):          # LIMIT may follow a newline, not just a space
        clean += f" LIMIT {MAX_ROWS}"                      # always cap output
    return clean


# ---------------------------------------------------------------------------
# STEP 9: LAYER 2C - QUERY EXECUTOR  (SQLite + Pandas, read-only connection)
# ---------------------------------------------------------------------------
def execute_sql(sql: str, db_path=DB_PATH) -> pd.DataFrame:
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)   # read-only = extra safety
    try:
        return pd.read_sql_query(sql, conn)
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# STEP 10: LAYER 2D - FINDINGS GENERATOR  (DataFrame -> structured finding, via templates)
# ---------------------------------------------------------------------------
def make_highlights(df: pd.DataFrame) -> list:
    """Deterministic highest/lowest per numeric column (the report must use these for superlatives)."""
    labels = [c for c in df.columns if not pd.api.types.is_numeric_dtype(df[c]) or c.endswith("_id")]
    nums = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c]) and not c.endswith("_id") and c != "n"]
    if len(df) < 3 or not labels or not nums:
        return []
    name = lambda i: " / ".join(str(df.loc[i, l]) for l in labels)
    out = []
    for c in nums:
        col = df[c].dropna()                               # NULLs (e.g. customers with no rated orders)
        if col.empty:
            continue
        out.append(f"(only among these {len(df)} rows) {c}: highest = {name(col.idxmax())} ({col.max()}), lowest = {name(col.idxmin())} ({col.min()})")
    return out


def make_finding(agent: str, sql: str, df: pd.DataFrame, source: str) -> dict:
    df = df.round(2)                                       # 19.79388... -> 19.79
    if df.empty:
        summary = f"{agent}: query returned no rows."
    else:
        rows = [", ".join(f"{c}={r[c]}" for c in df.columns) for r in df.head(6).to_dict("records")]
        summary = f"{agent}: {len(df)} rows. " + " | ".join(rows) + (" ..." if len(df) > 6 else "")
    return {"agent": agent, "sql": sql, "sql_source": source,
            "summary": summary, "highlights": make_highlights(df), "data": df.head(20).to_dict(orient="records")}


def check_result(task: dict, df: pd.DataFrame, sql: str = ""):
    """Guard: small LLMs often write SQL that runs but answers the wrong question.
    Returns None if OK, otherwise a short reason string."""
    if re.search(r"group\s+by[^;]*?\b(delivery_time|food_preparation_time|cost_of_the_order|rating)\b",
                 sql, flags=re.I | re.S):
        return "it GROUPs BY a measure column (delivery_time/rating/cost); group only by category columns"
    if re.search(r"rating_missing\s*(=\s*0|!=\s*1|<>\s*1)", sql, flags=re.I) and not re.search(r"\brating\b", sql, flags=re.I):
        return ("it filters rating_missing = 0 but the question is not about rating; "
                "remove that filter so ALL orders are counted")
    if re.search(r"\bhaving\b", sql, flags=re.I) and not re.search(r"\bavg\s*\(", sql, flags=re.I):
        return "remove HAVING: a minimum group size is only for AVERAGE rankings, not for counts, sums or top-N"
    # ranking customers/restaurants by an AVERAGE with no minimum group size -> one-order groups tie at the top
    m = re.search(r"order\s+by\s+([\w\.\(]+)", sql, flags=re.I)
    if m and re.search(r"group\s+by[^;]*\b(customer_id|restaurant_name)\b", sql, flags=re.I) \
            and not re.search(r"\bhaving\b", sql, flags=re.I):
        key = m.group(1)
        if key.lower().startswith("avg") or re.search(rf"avg\s*\(.*?\)\s+as\s+{re.escape(key)}\b", sql, flags=re.I | re.S):
            return ("it ranks customers/restaurants by an AVERAGE with no minimum group size; "
                    "add HAVING COUNT(*) >= 5 so one-order groups do not tie at the top")
    agent = DOMAIN_AGENTS[task["agent"]]
    if task.get("trend") and "trend_fallback" in agent:
        has_day_split = "day_of_the_week" in df.columns          # trend = Weekday vs Weekend columns
        entities = df.iloc[:, 0].nunique() if len(df.columns) else 0
        if not has_day_split:
            return "trend needs a day_of_the_week column in the SELECT and GROUP BY"
        if entities > (task.get("top_n") or 20):
            return "pick the top N entities in a subquery, then split by day_of_the_week (top N entities, not top N rows)"
    return None


def run_domain_agent(task: dict, schema_text: str, table: str, trace: Trace) -> dict:
    """Runs 2A -> 2B -> 2C -> 2D for one domain agent.
    If the SQL fails or answers the wrong question: ask the LLM to fix it once, then use the template."""
    agent = task["agent"]
    if agent == "overview":                                # fixed totals query, no LLM needed
        safe_sql = validate_sql(overview_sql(schema_text, table))
        df = execute_sql(safe_sql)
        trace.log("2C executed", agent=agent, rows=len(df))
        return make_finding(agent, safe_sql, df, "template")
    sql, source = generate_sql(task, schema_text, table)
    trace.log("2A sql_generation", agent=agent, source=source, sql=sql,
              llm_error=get_llm_error() if source != "llm" else None)
    df, safe_sql = None, None
    for attempt in range(2):
        try:
            safe_sql = validate_sql(fix_rating_filter(sql))     # 2B (+ auto-fix of the rating filter)
            df = execute_sql(safe_sql)                     # 2C
            reason = check_result(task, df, safe_sql)      # does the result really answer the question?
            if reason:
                raise ValueError(reason)
            break
        except Exception as e:
            trace.log("2B/2C rejected", agent=agent, attempt=attempt + 1, error=str(e))
            df = None
            if attempt == 0 and source == "llm":           # retry once, giving the model the reason
                sql, source = generate_sql(task, schema_text, table, feedback=str(e), prev_sql=sql)
                trace.log("2A sql_regenerated", agent=agent, source=source, sql=sql)
    if df is None:                                         # still bad -> safe template
        trace.log("2B/2C error_retry_fallback", agent=agent)
        safe_sql = validate_sql(get_fallback_sql(task, table))
        df, source = execute_sql(safe_sql), "fallback_template"
    trace.log("2C executed", agent=agent, rows=len(df))
    return make_finding(agent, safe_sql, df, source)       # 2D


# ---------------------------------------------------------------------------
# STEP 11: LAYER 3 - AGGREGATOR AGENT
# ---------------------------------------------------------------------------
def aggregate(findings: list) -> dict:
    seen, unique = {}, []
    for f in findings:
        key = str([list(row.values()) for row in f["data"]])   # same rows = same finding
        if key in seen:
            old = seen[key]["agent"]
            seen[key]["agent"] = f"{old} + {f['agent']}"    # merge duplicates (e.g. customer + revenue)
            seen[key]["summary"] = seen[key]["summary"].replace(f"{old}:", f"{seen[key]['agent']}:", 1)
        else:
            seen[key] = f
            unique.append(f)
    findings = unique
    return {"count": len(findings),
            "summaries": [f["summary"] for f in findings],
            "findings": findings}


# ---------------------------------------------------------------------------
# STEP 12: LAYER 4 - EXECUTIVE AGENT  (Qwen3 writes the report; template fallback)
# ---------------------------------------------------------------------------
def executive_report(question: str, unified: dict) -> str:
    system = ("You are a business analyst for a food delivery company. Using ONLY the findings given, "
              "write a short report: Key insights (bullets), then 3 recommendations. "
              "Use ONLY numbers that appear in the findings; do NOT add, divide, average or compute any new number. "
              "The data has no dates, so never claim an upward/downward trend over time; "
              "only Weekday vs Weekend comparisons are possible. "
              "When asked which is best/fastest, name the winner of THAT specific metric only if it is "
              "visible in the findings; if findings disagree or a group is small (n < 10), say so. "
              "For ANY best/worst/highest/lowest/fastest/slowest claim use ONLY the 'highlights' lists, "
              "and say which group they cover: highlights describe only the rows returned "
              "(e.g. the top 20 customers by spend), NEVER the whole business. "
              "Never mention Weekday/Weekend for a customer or restaurant unless that column is in the data. "
              "RECOMMENDATIONS: the data contains ONLY day_of_the_week (Weekday/Weekend), cuisine, restaurant, "
              "customer, order cost, rating, preparation time and delivery time. There is NO data on drivers, "
              "routes, staffing, time of day, menu prices, margins or promotions, so never base advice on those. "
              "Each recommendation must cite a number from the findings; give fewer than 3 if fewer are supported. "
              "If a group is small (a few orders), say the sample is small instead of drawing conclusions. "
              "If the question is a simple factual lookup (a count or one number), answer it in 1-2 sentences "
              "and skip the recommendations.")
    user = f"Question: {question}\nFindings:\n" + json.dumps([{"agent": f["agent"], "highlights": f["highlights"], "data": f["data"]}
                                for f in unified["findings"]], default=str)[:12000]
    report = call_llm(system, user, REPORT_MODEL)
    if report:
        return report
    # Fallback: simple report from the finding summaries
    return "KEY INSIGHTS\n" + "\n".join(f"- {s}" for s in unified["summaries"]) + \
           "\n\n(LLM unavailable: start Ollama with Qwen3 to get recommendations.)"


# ---------------------------------------------------------------------------
# STEP 13: LAYER 5 - MEMORY (future)  FAISS + sentence-transformers
# ---------------------------------------------------------------------------
def retrieve_memory(question: str) -> list:
    return []   # TODO: embed question -> search FAISS index of past Q&A -> return context


# ---------------------------------------------------------------------------
# STEP 14: ORCHESTRATOR  - runs the whole pipeline end to end
# ---------------------------------------------------------------------------
def json_safe(obj):
    """NaN/inf (from SQL NULLs) and numpy scalars are not valid JSON -> convert to None / plain Python."""
    if isinstance(obj, dict):
        return {k: json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [json_safe(v) for v in obj]
    if isinstance(obj, float):
        return None if (math.isnan(obj) or math.isinf(obj)) else float(obj)
    if hasattr(obj, "item"):                              # numpy int64 / float64 / bool_
        return json_safe(obj.item())
    return obj


def run_pipeline(question: str) -> dict:
    trace = Trace()

    clean_q = preprocess_query(question)                                  # Layer 0
    trace.log("0 preprocess", original=question, clean=clean_q)

    agents = classify_query(clean_q)                                      # Layer 0
    trace.log("0 classify", agents=agents)

    schema = read_schema()                                                # Layer 1
    table, schema_text = pick_main_table(schema), schema_to_text(schema)
    trace.log("1 schema", main_table=table)

    plan = plan_tasks(clean_q, agents)                                    # Layer 1
    trace.log("1 plan", tasks=[t["agent"] for t in plan])

    with ThreadPoolExecutor(max_workers=3) as pool:                              # Layer 2 (parallel)
        findings = list(pool.map(lambda t: run_domain_agent(t, schema_text, table, trace), plan))

    unified = aggregate(findings)                                         # Layer 3
    trace.log("3 aggregate", findings=unified["count"])

    report = executive_report(question, unified)                          # Layer 4
    trace.log("4 executive_report", chars=len(report))

    return json_safe({"question": question, "clean_question": clean_q, "agents": agents,
                      "report": report, "findings": unified["findings"], "trace": trace.to_json()})   # Layer 3A