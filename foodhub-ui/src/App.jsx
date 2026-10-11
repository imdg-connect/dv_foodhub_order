import React, { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  Area, Bar, BarChart, CartesianGrid, Cell, ComposedChart, Line, Pie, PieChart, ResponsiveContainer,
  Scatter, ScatterChart, Tooltip, XAxis, YAxis, ZAxis,
} from "recharts";
import {
  Activity, AlertTriangle, BarChart3, Bot, ChevronRight, Database, FileText, Info, Lightbulb, Loader2,
  Send, Sparkles, UtensilsCrossed,
} from "lucide-react";

const API_URL = "/api/ask";    // agents_api  -> http://127.0.0.1:8003/ask
const DASH_URL = "/dash";      // dashboard_api  (/start /stop /status)  -> http://127.0.0.1:8002
const DATA_URL = "/api";       // agents_api     (/orders, /ws SQLite)   -> http://127.0.0.1:8003
const STATUS_POLL_MS = 2000;

const PALETTE = ["#7c8cff", "#31e0c4", "#ff6fae", "#ffb454", "#5ec2ff", "#b58cff", "#9be564", "#ff8a65"];

const SUGGESTIONS = [
  "Which cuisines have the best ratings and fastest delivery?",
  "Give me an overall summary of the business",
  "Compare weekday vs weekend revenue",
  "Top 5 restaurants by orders",
  "Who are the top 10 customers by spend?",
];
const DEFAULT_QUESTION = SUGGESTIONS[0];   // answered automatically when the page loads

/* ------------------------------------------------------------------ helpers */
const fmt = (n, d = 0) =>
  n == null || Number.isNaN(n) ? "–" : Number(n).toLocaleString(undefined, { maximumFractionDigits: d, minimumFractionDigits: d });
const isNum = (v) => typeof v === "number" && Number.isFinite(v);
const short = (s, n = 16) => (String(s).length > n ? `${String(s).slice(0, n - 1)}…` : String(s));
const AXIS = { fill: "#8b92b0", fontSize: 10 };

function Tip({ active, payload, label }) {
  if (!active || !payload?.length) return null;
  return (
    <div className="tip">
      <b>{label ?? payload[0].name}</b>
      {payload.map((p, i) => (
        <div key={i} style={{ color: p.color || p.payload?.fill }}>
          {p.name}: {fmt(p.value, 2)}
        </div>
      ))}
    </div>
  );
}

function ScatterTip({ active, payload }) {
  if (!active || !payload?.length) return null;
  const p = payload[0].payload;
  return (
    <div className="tip">
      <b>{p.name}</b>
      <div>Avg delivery: {fmt(p.x, 1)} min</div>
      <div>Avg rating: {fmt(p.y, 2)}</div>
      <div>Orders: {fmt(p.z)}</div>
    </div>
  );
}

function inline(text) {
  return text.split(/(\*\*[^*]+\*\*)/g).map((part, i) =>
    part.startsWith("**") ? <b key={i}>{part.slice(2, -2)}</b> : <React.Fragment key={i}>{part}</React.Fragment>
  );
}

/* ------------------------------------------------------------------ report (left, 40%) */
const SECTION_META = {
  "KEY INSIGHTS": { title: "Summary", Icon: FileText, cls: "summary", ordered: false },
  RECOMMENDATIONS: { title: "Recommendations", Icon: Lightbulb, cls: "reco", ordered: true },
  NOTES: { title: "Notes", Icon: Info, cls: "notes", ordered: false },
  ANSWER: { title: "Answer", Icon: Sparkles, cls: "summary", ordered: false },
};

function parseReport(text) {
  const secs = [];
  let cur = null;
  (text || "").split("\n").forEach((raw) => {
    const line = raw.trim();
    if (!line) return;
    if (/^[A-Z][A-Z\s]{4,}$/.test(line)) { cur = { name: line, items: [] }; secs.push(cur); return; }
    if (!cur) { cur = { name: "ANSWER", items: [] }; secs.push(cur); }
    cur.items.push(line.replace(/^([-*•]|\d+[.)])\s+/, "").replace(/^ANSWER:\s*/i, ""));
  });
  return secs;
}

function Section({ name, items }) {
  const meta = SECTION_META[name] || {
    title: name.charAt(0) + name.slice(1).toLowerCase(), Icon: FileText, cls: "summary", ordered: false,
  };
  const List = meta.ordered ? "ol" : "ul";
  return (
    <div className={`plastic card sec ${meta.cls}`}>
      <h3><meta.Icon size={15} /> {meta.title}</h3>
      <List>{items.map((t, i) => <li key={i}>{inline(t)}</li>)}</List>
    </div>
  );
}

function ReportPanel({ state }) {
  const { loading, error, data } = state;

  if (loading) {
    return (
      <div className="plastic card">
        <h3><Loader2 size={15} className="spin" /> Agents are working…</h3>
        {[90, 70, 85, 55, 78].map((w, i) => <div key={i} className="skeleton" style={{ width: `${w}%` }} />)}
      </div>
    );
  }
  if (error) {
    return (
      <div className="plastic card error">
        <h3><AlertTriangle size={15} /> Request failed</h3>
        <div style={{ fontSize: 14 }}>{error}</div>
        <div style={{ fontSize: 12, color: "#8b92b0", marginTop: 8 }}>
          Check that the API is running
        </div>
      </div>
    );
  }
  if (!data) {
    return (
      <div className="plastic card empty">
        <Bot size={34} />
        <div>Ask anything about orders, cuisines, restaurants, customers, ratings or delivery times.</div>
      </div>
    );
  }

  const sections = parseReport(data.report);
  const highlights = [...new Set((data.findings || []).flatMap((f) => f.highlights || []))];
  const trace = data.trace || [];
  return (
    <>
      <div className="tags" style={{ marginBottom: 0 }}>
        {(data.agents || []).map((a) => <span key={a} className="tag">{a}</span>)}
        {data.clean_question && data.clean_question !== data.question?.toLowerCase() && (
          <span className="tag amber">interpreted: {data.clean_question}</span>
        )}
      </div>

      {sections.map((s, i) => <Section key={i} {...s} />)}
      {highlights.length > 0 && <Section name="HIGHLIGHTS" items={highlights} />}

      {trace.length > 0 && (
        <div className="plastic card">
          <h3><Activity size={15} /> Pipeline trace</h3>
          <div className="timeline">
            {trace.map((t, i) => {
              const { step, time, ...rest } = t;
              const bad = /rejected|error/i.test(step);
              const extra = Object.entries(rest).filter(([k]) => k !== "sql" && rest[k] != null)
                .map(([k, v]) => `${k}=${typeof v === "object" ? JSON.stringify(v) : v}`).join("  ");
              return (
                <div key={i} className={`tl ${bad ? "bad" : ""}`}>
                  <i /><span><b style={{ color: "#e9ecf8" }}>{step}</b> · {time} {extra && <>· {extra}</>}</span>
                </div>
              );
            })}
          </div>
        </div>
      )}
    </>
  );
}

/* ------------------------------------------------------------------ generated graphs (middle, 30%) */
const COUNT_COLS = new Set(["n", "rated_orders"]);

function FindingCard({ f }) {
  const rows = f.data || [];
  const cols = rows.length ? Object.keys(rows[0]) : [];
  let nums = cols.filter((c) => rows.some((r) => isNum(r[c])) && !c.endsWith("_id") && !COUNT_COLS.has(c));
  let labels = cols.filter((c) => !nums.includes(c) && !COUNT_COLS.has(c));
  if (!labels.length && nums.includes("rating") && nums.length > 1) {      // rating distribution
    labels = ["rating"];
    nums = nums.filter((c) => c !== "rating");
  }
  const [metric, setMetric] = useState(nums[0]);
  const active = nums.includes(metric) ? metric : nums[0];
  const color = PALETTE[Math.max(0, nums.indexOf(active)) % PALETTE.length];

  const chartData = rows.map((r, i) => ({ ...r, __label: labels.map((l) => r[l]).join(" · ") || `#${i + 1}` }));
  const single = rows.length === 1;
  const MAX_BARS = 8;                                         // fixed-size chart: top N bars, full list in the data table
  const horizontal = chartData.length > 5;
  const bars = horizontal ? chartData.slice(0, MAX_BARS) : chartData;
  const height = horizontal ? 220 : 170;

  return (
    <div className="plastic card">
      <h3><BarChart3 size={15} /> {String(f.agent).replace(/ \+ /g, " + ")}</h3>
      <div className="tags">
        <span className="tag">{rows.length} rows</span>
        <span className={`tag ${f.sql_source === "llm" ? "green" : "amber"}`}>
          {f.sql_source === "llm" ? "LLM SQL" : f.sql_source === "template" ? "template SQL" : "compiled SQL"}
        </span>
      </div>

      {single ? (
        <div className="kpis" style={{ gridTemplateColumns: "repeat(auto-fit,minmax(110px,1fr))" }}>
          {cols.map((c) => (
            <div key={c} className="plastic kpi">
              <small>{c.replace(/_/g, " ")}</small>
              <div style={{ fontSize: 18 }}>{isNum(rows[0][c]) ? fmt(rows[0][c], 2) : String(rows[0][c])}</div>
            </div>
          ))}
        </div>
      ) : active && rows.length > 1 && (
        <>
          {nums.length > 1 && (
            <div className="mchips">
              {nums.map((c) => (
                <button key={c} className={`mchip ${c === active ? "on" : ""}`} onClick={() => setMetric(c)}>
                  {c.replace(/_/g, " ")}
                </button>
              ))}
            </div>
          )}
          <div style={{ height }}>
            <ResponsiveContainer>
              {horizontal ? (
                <BarChart data={bars} layout="vertical" margin={{ top: 4, right: 12, left: 0, bottom: 0 }}>
                  <CartesianGrid stroke="rgba(255,255,255,.06)" horizontal={false} />
                  <XAxis type="number" tick={AXIS} />
                  <YAxis type="category" dataKey="__label" width={104} interval={0} tick={{ ...AXIS, fill: "#c9cde6" }}
                    tickFormatter={(v) => short(v, 17)} />
                  <Tooltip content={<Tip />} cursor={{ fill: "rgba(255,255,255,.05)" }} />
                  <Bar dataKey={active} fill={color} radius={[0, 6, 6, 0]} />
                </BarChart>
              ) : (
                <BarChart data={bars} margin={{ top: 8, right: 8, left: -14, bottom: 0 }}>
                  <CartesianGrid stroke="rgba(255,255,255,.06)" vertical={false} />
                  <XAxis dataKey="__label" tick={AXIS} interval={0} tickFormatter={(v) => short(v, 11)} />
                  <YAxis tick={AXIS} />
                  <Tooltip content={<Tip />} cursor={{ fill: "rgba(255,255,255,.05)" }} />
                  <Bar dataKey={active} fill={color} radius={[6, 6, 0, 0]} />
                </BarChart>
              )}
            </ResponsiveContainer>
          </div>
          {bars.length < chartData.length && (
            <div className="hint" style={{ padding: "2px 6px 6px" }}>
              Top {bars.length} of {chartData.length} shown · full list in the data table
            </div>
          )}
        </>
      )}

      {rows.length > 0 && (
        <details>
          <summary>Data table ({rows.length} rows)</summary>
          <div className="tablewrap" style={{ marginTop: 8 }}>
            <table>
              <thead><tr>{cols.map((c) => <th key={c}>{c}</th>)}</tr></thead>
              <tbody>{rows.map((r, i) => <tr key={i}>{cols.map((c) => <td key={c}>{r[c] == null ? "–" : String(r[c])}</td>)}</tr>)}</tbody>
            </table>
          </div>
        </details>
      )}
      <details><summary>SQL</summary><pre>{f.sql}</pre></details>
    </div>
  );
}

function GraphsPanel({ state }) {
  const { loading, error, data } = state;
  if (loading) {
    return (
      <div className="plastic card">
        <h3><Loader2 size={15} className="spin" /> Building graphs…</h3>
        {[80, 60, 90, 50].map((w, i) => <div key={i} className="skeleton" style={{ width: `${w}%`, height: 28 }} />)}
      </div>
    );
  }
  if (error || !data) {
    return <div className="plastic card empty"><BarChart3 size={30} /><div>Graphs for your question appear here.</div></div>;
  }
  if (!(data.findings || []).length) {
    return <div className="plastic card empty"><div>No graph data for this question.</div></div>;
  }
  return <>{data.findings.map((f, i) => <FindingCard key={`${f.agent}-${i}`} f={f} />)}</>;
}

/* ------------------------------------------------------------------ live data hook (right, 30%) */
// dashboard_api.status() -> feed.status(); accept a few likely shapes
const isRunning = (s) => {
  if (s?.stopping) return false;   // stop was called: show OFF right away while the last EDA finishes
  return Boolean(s?.running ?? s?.live_feed ?? s?.status === "running");
};
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

function useDashboard() {
  const [rows, setRows] = useState([]);
  const [updated, setUpdated] = useState(null);
  const [err, setErr] = useState(null);
  const [dataErr, setDataErr] = useState(null);     // problem loading SQLite rows
  const [loaded, setLoaded] = useState(false);
  const [running, setRunning] = useState(false);    // from dashboard_api GET /status
  const [busy, setBusy] = useState(false);

  const readStatus = useCallback(async () => {
    const res = await fetch(`${DASH_URL}/status`);
    if (res.status === 404) throw new Error("no /status on :8002 – wrong app or old dashboard_api process");
    if (!res.ok) throw new Error(`status ${res.status}`);
    const r = isRunning(await res.json());
    setRunning(r);
    return r;
  }, []);

  // initial data from SQLite + keep status in sync
  useEffect(() => {
    fetch(`${DATA_URL}/orders`)
      .then(async (r) => {
        if (!r.ok) {
          let d = ""; try { d = (await r.json()).detail; } catch {}
          throw new Error(d || `agents_api returned ${r.status} (is it running on :8003?)`);
        }
        return r.json();
      })
      .then((d) => { setRows(d); setUpdated(new Date()); setDataErr(null); setLoaded(true); })
      .catch((e) => setDataErr(e.message));
     readStatus().catch((e) => setErr(e.message));      // ONE status check on page load only
  }, [readStatus]); 
  useEffect(() => {
    if (!running) return;                              // toggle OFF -> no polling
    const id = setInterval(() => readStatus().catch(() => {}), STATUS_POLL_MS);
    return () => clearInterval(id);
  }, [running, readStatus]);

  // websocket: server pushes rows whenever the SQLite data changes (auto-reconnect)
    // websocket: open ONLY while the toggle is ON (running = true); closed otherwise
  useEffect(() => {
    if (!running) return;                            // toggle OFF -> no websocket at all
    let ws, timer, closed = false;
    const connect = () => {
      const proto = location.protocol === "https:" ? "wss" : "ws";
      ws = new WebSocket(`${proto}://${location.host}${DATA_URL}/ws`);
      ws.onmessage = (ev) => {
        const msg = JSON.parse(ev.data);
        if (msg.type === "orders") { setRows(msg.rows); setUpdated(new Date()); setDataErr(null); setLoaded(true); }
        if (msg.type === "error") setDataErr(msg.detail);
      };
      ws.onclose = () => { if (!closed) timer = setTimeout(connect, 2000); };
      ws.onerror = () => ws.close();
    };
    connect();
    return () => {                                   // toggle just went OFF
      closed = true;                                 // stop auto-reconnect
      clearTimeout(timer);
      const last = ws;
      setTimeout(() => {                             // 10 s grace so the last EDA's rows can arrive
        last?.close();
        fetch(`${DATA_URL}/orders`)                  // one final refresh, then silence
          .then((r) => (r.ok ? r.json() : null))
          .then((d) => { if (d) { setRows(d); setUpdated(new Date()); } })
          .catch(() => {});
      }, 10000);
    };
  }, [running]);

  // toggle: ON -> POST /start, OFF -> POST /stop, then wait for status() to confirm
  // (after /stop the current EDA run may still be finishing, so status can lag)
  const toggle = useCallback(async () => {
    const want = !running;
    setBusy(true);
    try {
      let res;
      for (let i = 0; i < 10; i++) {              // retry while the server says "still finishing the last run"
        res = await fetch(`${DASH_URL}/${want ? "start" : "stop"}`, { method: "POST" });
        if (res.status !== 409) break;
        await sleep(1000);
      }
      
      if (!res.ok) {
        let detail = ""; try { detail = (await res.json()).detail; } catch {}
        throw new Error(detail || (res.status === 404 ? "no /start or /stop on :8002 – wrong app or old process" : `${want ? "start" : "stop"} ${res.status}`));
      }
      for (let i = 0; i < 60; i++) {            // up to ~60 s
        if ((await readStatus()) === want) break;
        await sleep(1000);
      }
      setErr(null);
    } catch (e) { setErr(e.message); }
    setBusy(false);
  }, [running, readStatus]);

  return { rows, updated, err, dataErr, loaded, running, busy, toggle };
}

const group = (rows, key) => {
  const m = new Map();
  rows.forEach((r) => {
    const k = r[key];
    if (!m.has(k)) m.set(k, []);
    m.get(k).push(r);
  });
  return m;
};
const sum = (a, f) => a.reduce((s, r) => s + (Number(r[f]) || 0), 0);
const avg = (a, f) => (a.length ? sum(a, f) / a.length : 0);

/* ------------------------------------------------------------------ compact dashboard cards */
function PieCard({ title, data, donut = true }) {
  const total = data.reduce((s, d) => s + d.value, 0);
  return (
    <div className="plastic chart">
      <h3>{title}</h3>
      <div style={{ height: 120 }}>
        <ResponsiveContainer>
          <PieChart>
            <Pie data={data} dataKey="value" nameKey="name" innerRadius={donut ? 30 : 0} outerRadius={52} paddingAngle={2} stroke="none">
              {data.map((d, i) => <Cell key={d.name} fill={PALETTE[i % PALETTE.length]} />)}
            </Pie>
            <Tooltip content={<Tip />} />
          </PieChart>
        </ResponsiveContainer>
      </div>
      <div className="legend">
        {data.slice(0, 4).map((d, i) => (
          <span key={d.name}><i style={{ background: PALETTE[i % PALETTE.length] }} />{short(d.name, 12)} {total ? Math.round((d.value / total) * 100) : 0}%</span>
        ))}
      </div>
    </div>
  );
}

function BarCard({ title, data, dataKey, color, horizontal = false, span2 = false, height = 140, colorful = false }) {
  return (
    <div className={`plastic chart ${span2 ? "span2" : ""}`}>
      <h3>{title}</h3>
      <div style={{ height }}>
        <ResponsiveContainer>
          <BarChart data={data} layout={horizontal ? "vertical" : "horizontal"} margin={{ top: 4, right: 10, left: horizontal ? 0 : -22, bottom: 0 }}>
            <CartesianGrid stroke="rgba(255,255,255,.06)" vertical={horizontal} horizontal={!horizontal} />
            {horizontal ? (
              <>
                <XAxis type="number" tick={AXIS} />
                <YAxis type="category" dataKey="name" width={88} interval={0} tick={{ ...AXIS, fill: "#c9cde6" }} tickFormatter={(v) => short(v, 14)} />
              </>
            ) : (
              <>
                <XAxis dataKey="name" tick={AXIS} interval={0} />
                <YAxis tick={AXIS} />
              </>
            )}
            <Tooltip content={<Tip />} cursor={{ fill: "rgba(255,255,255,.05)" }} />
            <Bar dataKey={dataKey} fill={color} radius={horizontal ? [0, 5, 5, 0] : [5, 5, 0, 0]}>
              {colorful && data.map((d, i) => <Cell key={i} fill={PALETTE[i % PALETTE.length]} />)}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}

function TrendCard({ hist, running }) {
  return (
    <div className="plastic chart span2">
      <h3>Live trend · orders &amp; avg delivery</h3>
      {hist.length < 2 ? (
        <div className="hint">{running ? "Waiting for the next live update…" : "Turn on live data to build the trend."}</div>
      ) : (
        <>
          <div style={{ height: 140 }}>
            <ResponsiveContainer>
              <ComposedChart data={hist} margin={{ top: 6, right: 4, left: -22, bottom: 0 }}>
                <CartesianGrid stroke="rgba(255,255,255,.06)" vertical={false} />
                <XAxis dataKey="t" tick={AXIS} minTickGap={28} />
                <YAxis yAxisId="l" tick={AXIS} domain={["dataMin - 1", "dataMax + 1"]} allowDecimals={false} />
                <YAxis yAxisId="r" orientation="right" tick={AXIS} domain={["dataMin - 0.2", "dataMax + 0.2"]} />
                <Tooltip content={<Tip />} />
                <Area yAxisId="l" type="monotone" dataKey="orders" name="orders" stroke={PALETTE[0]} fill={PALETTE[0]} fillOpacity={0.2} isAnimationActive={false} />
                <Line yAxisId="r" type="monotone" dataKey="delivery" name="avg delivery (min)" stroke={PALETTE[1]} strokeWidth={2} dot={false} isAnimationActive={false} />
              </ComposedChart>
            </ResponsiveContainer>
          </div>
          <div className="legend">
            <span><i style={{ background: PALETTE[0] }} />orders</span>
            <span><i style={{ background: PALETTE[1] }} />avg delivery (min)</span>
          </div>
        </>
      )}
    </div>
  );
}

function ScatterCard({ title, data }) {
  return (
    <div className="plastic chart span2">
      <h3>{title}</h3>
      <div style={{ height: 170 }}>
        <ResponsiveContainer>
          <ScatterChart margin={{ top: 8, right: 14, left: -16, bottom: 14 }}>
            <CartesianGrid stroke="rgba(255,255,255,.06)" />
            <XAxis type="number" dataKey="x" name="delivery" tick={AXIS} domain={[(v) => Math.floor(v - 1), (v) => Math.ceil(v + 1)]}
              label={{ value: "avg delivery (min)", position: "insideBottom", offset: -8, fill: "#8b92b0", fontSize: 10 }} />
            <YAxis type="number" dataKey="y" name="rating" tick={AXIS} domain={[(v) => +(v - 0.1).toFixed(1), (v) => +(v + 0.1).toFixed(1)]} />
            <ZAxis type="number" dataKey="z" range={[40, 240]} />
            <Tooltip content={<ScatterTip />} cursor={{ strokeDasharray: "3 3" }} />
            <Scatter data={data}>
              {data.map((d, i) => <Cell key={d.name} fill={PALETTE[i % PALETTE.length]} fillOpacity={0.85} />)}
            </Scatter>
          </ScatterChart>
        </ResponsiveContainer>
      </div>
      <div className="hint" style={{ padding: "0 8px 6px" }}>Top-left = faster &amp; better rated · bubble size = orders</div>
    </div>
  );
}

function SummaryTable({ m }) {
  const pct = (a, b) => (b ? `${fmt((a / b) * 100)}%` : "–");
  const rowsT = [
    ["Orders", fmt(m.orders), `${fmt(m.customers)} customers · ${fmt(m.restaurants)} restaurants`],
    ["Cuisines", fmt(m.cuisines), m.topCuisine ? `most orders: ${m.topCuisine}` : ""],
    ["Revenue", `$${fmt(m.revenue)}`, `$${fmt(m.revenue / m.orders, 2)} per order`],
    m.hasCommission && ["Commission", `$${fmt(m.commission)}`, `${pct(m.commission, m.revenue)} of revenue`],
    ["Avg delivery", `${fmt(m.delivery, 1)} min`, `prep ${fmt(m.prep, 1)} min`],
    ["Avg rating", fmt(m.rating, 2), `${fmt(m.ratedCount)} rated (${pct(m.ratedCount, m.orders)})`],
    ["Weekend / weekday", `${fmt(m.weekend)} / ${fmt(m.orders - m.weekend)}`, `${pct(m.weekend, m.orders)} weekend`],
    m.hasSla && ["SLA breached", fmt(m.breached), `${pct(m.breached, m.orders)} of orders`],
  ].filter(Boolean);
  return (
    <div className="plastic chart">
      <h3><Database size={13} /> Data at a glance</h3>
      <table className="sumtable">
        <tbody>
          {rowsT.map(([k, v, d]) => (
            <tr key={k}><td>{k}</td><td className="v">{v}</td><td className="d">{d}</td></tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function computeMetrics(rows) {
    if (!rows.length) return null;
    const has = (c) => c in rows[0];
    const isRated = (r) => (has("rating_missing") ? r.rating_missing === 0 : isNum(r.rating));
    const rated = rows.filter(isRated);

    const byCuisine = [...group(rows, "cuisine_type")].map(([name, a]) => {
      const ra = a.filter(isRated);
      return {
        name: String(name), orders: a.length, revenue: sum(a, "cost_of_the_order"),
        delivery: +avg(a, "delivery_time").toFixed(1), rating: ra.length ? +avg(ra, "rating").toFixed(2) : 0,
      };
    }).sort((a, b) => b.orders - a.orders);

    const cuisineBars = byCuisine.slice(0, 6).map((c) => ({ name: c.name, orders: c.orders }));
    const rest = byCuisine.slice(6).reduce((s, c) => s + c.orders, 0);
    if (rest) cuisineBars.push({ name: "Other", orders: rest });

    const pieOf = (key) => (has(key)
      ? [...group(rows, key)].map(([name, a]) => ({ name: String(name).replace(/_/g, " "), value: a.length })).sort((a, b) => b.value - a.value)
      : []);

    const restaurants = [...group(rows, "restaurant_name")].map(([name, a]) => ({ name: String(name), orders: a.length }))
      .sort((a, b) => b.orders - a.orders).slice(0, 6);

    const bands = has("price_band")
      ? [...group(rows, "price_band")].map(([name, a]) => ({ name: String(name).replace(/_/g, " "), revenue: +sum(a, "cost_of_the_order").toFixed(0) }))
          .sort((a, b) => b.revenue - a.revenue)
      : [];

    const dh = {};
    rows.forEach((r) => { if (isNum(r.delivery_time)) { const b = Math.floor(r.delivery_time / 5) * 5; dh[b] = (dh[b] || 0) + 1; } });
    const deliveryHist = Object.keys(dh).map(Number).sort((a, b) => a - b).map((b) => ({ name: `${b}-${b + 4}`, orders: dh[b] }));

    const rc = {};
    rated.forEach((r) => { if (isNum(r.rating)) { const k = Math.round(r.rating); rc[k] = (rc[k] || 0) + 1; } });
    const ratingDist = Object.keys(rc).map(Number).sort((a, b) => a - b).map((k) => ({ name: `${k}★`, orders: rc[k] }));

    const scatter = byCuisine.filter((c) => c.rating > 0 && c.orders >= 5).map((c) => ({ name: c.name, x: c.delivery, y: c.rating, z: c.orders }));

    const weekend = rows.filter((r) => String(r.day_of_the_week).toLowerCase().includes("weekend")).length;
    const breached = has("total_time_sla") ? rows.filter((r) => String(r.total_time_sla).toUpperCase() === "BREACHED").length : 0;

    return {
      orders: rows.length,
      revenue: sum(rows, "cost_of_the_order"),
      hasCommission: has("commission"), commission: sum(rows, "commission"),
      delivery: avg(rows, "delivery_time"), prep: avg(rows, "food_preparation_time"),
      rating: avg(rated, "rating"), ratedCount: rated.length,
      customers: new Set(rows.map((r) => r.customer_id)).size,
      restaurants: new Set(rows.map((r) => r.restaurant_name)).size,
      cuisines: byCuisine.length, topCuisine: byCuisine[0]?.name,
      weekend, hasSla: has("total_time_sla"), breached,
      cuisineBars, dayPie: pieOf("day_of_the_week"), slaPie: pieOf("total_time_sla"), segPie: pieOf("customer_segment"),
      restaurantBars: restaurants, bands, deliveryHist, ratingDist, scatter,
    };
}

/* compact 2x2 block shown in the middle column */
function Distributions({ m }) {
  if (!m) return null;
  return (
    <div>
      <div className="sec-title">Live distributions</div>
      <div className="grid">
        <BarCard title="Delivery time (min)" data={m.deliveryHist} dataKey="orders" color={PALETTE[1]} height={100} />
        <BarCard title="Rating distribution" data={m.ratingDist} dataKey="orders" color={PALETTE[3]} height={100} />
        {m.bands.length > 0 && <BarCard title="Revenue by price band ($)" data={m.bands} dataKey="revenue" color={PALETTE[2]} horizontal height={100} />}
        {m.segPie.length > 0 && <PieCard title="Customer segments" data={m.segPie} />}
      </div>
    </div>
  );
}

function Dashboard({ dash, m }) {
  const { rows, updated, err, dataErr, loaded, running, busy, toggle } = dash;
  const [hist, setHist] = useState([]);

  // session trend: one point every time fresh rows arrive
  useEffect(() => {
    if (!rows.length) return;
    const t = new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" });
    setHist((h) => [...h, { t, orders: rows.length, delivery: +avg(rows, "delivery_time").toFixed(2) }].slice(-40));
  }, [rows]);


  return (
    <>
      <div className="dash-head">
        <h2>Live Business Dashboard</h2>
        <div className="live-wrap">
          <span className={`live-text ${running ? "on" : "off"}`}>
            {err ? `offline (${err})` : running ? `Live data running${updated ? ` · ${updated.toLocaleTimeString()}` : ""}` : "Live data stopped"}
          </span>
          <button
            className={`switch ${running ? "on" : "off"}`}
            onClick={toggle}
            disabled={busy}
            role="switch"
            aria-checked={running}
            aria-label="Toggle live data"
            title={running ? "Click to stop live data" : "Click to start live data"}
          >
            <i />
          </button>
        </div>
      </div>

      
      {!m ? (
        dataErr ? (
          <div className="plastic card error"><h3><AlertTriangle size={15} /> Can't load orders</h3><div style={{ fontSize: 14 }}>{dataErr}</div></div>
        ) : loaded ? (
          <div className="plastic card empty"><div>Connected, but the SQLite table has 0 rows.</div></div>
        ) : (
          <div className="plastic card empty"><Loader2 className="spin" size={28} /><div>Loading orders from SQLite…</div></div>
        )
      ) : (
        <div className="scroll col" style={{ flex: 1 }}>
          <SummaryTable m={m} />
          <div className="grid">
            <TrendCard hist={hist} running={running} />
            <BarCard title="Orders by cuisine" data={m.cuisineBars} dataKey="orders" color={PALETTE[0]} horizontal span2 height={150} colorful />
            <PieCard title="Weekday vs weekend" data={m.dayPie} />
            {m.slaPie.length > 0 && <PieCard title="Total-time SLA" data={m.slaPie} />}
            {m.scatter.length > 1 && <ScatterCard title="Cuisines · delivery vs rating" data={m.scatter} />}
            <BarCard title="Top restaurants by orders" data={m.restaurantBars} dataKey="orders" color={PALETTE[4]} horizontal span2 height={150} />
          </div>
        </div>
      )}
    </>
  );
}

/* ------------------------------------------------------------------ app */
export default function App() {
  const [q, setQ] = useState(DEFAULT_QUESTION);
  const [state, setState] = useState({ loading: false, error: null, data: null });
  const abortRef = useRef(null);
  const dash = useDashboard();                                   // live rows shared by middle + right columns
  const m = useMemo(() => computeMetrics(dash.rows), [dash.rows]);

  const submit = useCallback(async (question) => {
    const text = (question ?? q).trim();
    if (!text) return;
    abortRef.current?.abort();
    const ctrl = new AbortController();
    abortRef.current = ctrl;
    setState({ loading: true, error: null, data: null });
    try {
      const res = await fetch(API_URL, {
        method: "POST",
        headers: { "Content-Type": "application/json", accept: "application/json" },
        body: JSON.stringify({ question: text }),
        signal: ctrl.signal,
      });
      if (!res.ok) throw new Error(`API returned ${res.status}: ${(await res.text()).slice(0, 200)}`);
      setState({ loading: false, error: null, data: await res.json() });
    } catch (e) {
      if (e.name === "AbortError") return;
      setState({ loading: false, error: e.message, data: null });
    }
  }, [q]);

  // page load: always answer the preselected question
  useEffect(() => { submit(DEFAULT_QUESTION); }, []); // eslint-disable-line react-hooks/exhaustive-deps

  return (
    <div className="app">
      {/* 40% · question + written answer */}
      <section className="col" style={{ minHeight: 0 }}>
        <div className="brand">
          <div className="brand-badge"><UtensilsCrossed size={22} color="#fff" /></div>
          <div><h1>FoodHub Command Center</h1><p>Multi-agent analytics · ask in plain English</p></div>
        </div>

        <div className="plastic ask">
          <textarea
            value={q}
            placeholder="Ask a question about your food delivery data…"
            onChange={(e) => setQ(e.target.value)}
            onKeyDown={(e) => { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); submit(); } }}
          />
          <button className="orb" onClick={() => submit()} disabled={state.loading} title="Run query (Enter)" aria-label="Run query">
            {state.loading ? <Loader2 size={24} className="spin" /> : <Send size={24} />}
          </button>
        </div>

        <div className="chips">
          {SUGGESTIONS.map((s) => (
            <button key={s} className="chip" onClick={() => { setQ(s); submit(s); }}>
              <ChevronRight size={11} style={{ verticalAlign: "-1px" }} /> {s}
            </button>
          ))}
        </div>

        <div className="scroll col" style={{ flex: 1 }}>
          <ReportPanel state={state} />
        </div>
      </section>

      {/* 30% · graphs generated for the question */}
      <section className="col" style={{ minHeight: 0 }}>
        <div className="dash-head"><h2>Generated graphs</h2></div>
        <div className="scroll col" style={{ flex: 1 }}>
          <GraphsPanel state={state} />
          <Distributions m={m} />
        </div>
      </section>

      {/* 30% · live dashboard */}
      <section className="col" style={{ minHeight: 0 }}>
        <Dashboard dash={dash} m={m} />
      </section>
    </div>
  );
}