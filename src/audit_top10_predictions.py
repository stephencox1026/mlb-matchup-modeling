#!/usr/bin/env python3
"""
Post-hoc audit: league-wide top-10 HR / Hit / XBH vs same-day full-game Statcast outcomes.

Reads archived predictions (data/reports/archive/*/todays_matchup_predictions.json),
does not train or alter models.

Outputs:
  - data/reports/top10_prediction_audit.md
  - data/reports/top10_prediction_audit.csv
  - data/reports/top10_prediction_audit.html
"""
from __future__ import annotations

import csv
import json
import math
import re
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
ARCHIVE_DIR = ROOT / "data" / "reports" / "archive"
PA_PATH = ROOT / "data" / "raw" / "statcast_pa_level_league.parquet"
OUT_MD = ROOT / "data" / "reports" / "top10_prediction_audit.md"
OUT_CSV = ROOT / "data" / "reports" / "top10_prediction_audit.csv"
OUT_HTML = ROOT / "data" / "reports" / "top10_prediction_audit.html"

HIT_EVENTS = {"single", "double", "triple", "home_run"}
XBH_EVENTS = {"double", "triple", "home_run"}


def slate_date_from_path(path: Path) -> str | None:
    parts = path.resolve().parts
    for i, p in enumerate(parts):
        if p == "archive" and i + 1 < len(parts) and re.fullmatch(r"\d{4}-\d{2}-\d{2}", parts[i + 1]):
            return parts[i + 1]
    return None


def load_batter_day_stats() -> pd.DataFrame:
    pa = pd.read_parquet(PA_PATH, columns=["game_date", "batter", "events"])
    pa["game_date"] = pd.to_datetime(pa["game_date"], errors="coerce").dt.normalize()
    pa = pa.dropna(subset=["game_date", "batter"])
    pa["batter"] = pa["batter"].astype(int)
    ev = pa["events"].fillna("").str.lower()
    pa["is_hit"] = ev.isin(HIT_EVENTS).astype(int)
    pa["is_hr"] = (ev == "home_run").astype(int)
    pa["is_xbh"] = ev.isin(XBH_EVENTS).astype(int)
    g = pa.groupby(["game_date", "batter"], as_index=False).agg(
        PA=("events", "count"),
        H=("is_hit", "sum"),
        HR=("is_hr", "sum"),
        XBH=("is_xbh", "sum"),
    )
    return g


def top10_unique(df: pd.DataFrame, sort_col: str) -> pd.DataFrame:
    s = df.sort_values(sort_col, ascending=False, na_position="last")
    s = s.drop_duplicates(subset=["batter_mlbam_id"], keep="first")
    return s.head(10)


def matchup_label(row) -> str:
    return f"{row['batter_name']} ({row['batter_team']}) vs {row['pitcher_name']}"


def lerp_rgba(t: float, alpha: float = 0.5) -> str:
    """Orange (low) -> blue (high), t in [0,1]."""
    t = max(0.0, min(1.0, t))
    r1, g1, b1 = 255, 140, 0
    r2, g2, b2 = 37, 99, 235
    r = int(r1 + (r2 - r1) * t)
    g = int(g1 + (g2 - g1) * t)
    b = int(b1 + (b2 - b1) * t)
    return f"rgba({r},{g},{b},{alpha})"


def heat_style(val, mn: float, mx: float) -> str:
    if val is None or (isinstance(val, float) and math.isnan(val)):
        return ""
    if mx <= mn:
        t = 0.5
    else:
        t = (float(val) - mn) / (mx - mn)
    return f"background:{lerp_rgba(t)};"


def collect_rows(stats: pd.DataFrame, preds: pd.DataFrame, slate: str, list_type: str, sort_col: str) -> list[dict]:
    use = preds[(preds["pitcher_mlbam_id"].fillna(0).astype(int) > 0)].copy()
    if use.empty:
        return []
    top = top10_unique(use, sort_col)
    rows = []
    sdt = pd.Timestamp(slate).normalize()
    for rank, (_, r) in enumerate(top.iterrows(), 1):
        bid = int(r["batter_mlbam_id"])
        sub = stats[(stats["game_date"] == sdt) & (stats["batter"] == bid)]
        if sub.empty:
            pa, h, hr, xbh = 0, 0, 0, 0
        else:
            row0 = sub.iloc[0]
            pa, h, hr, xbh = int(row0["PA"]), int(row0["H"]), int(row0["HR"]), int(row0["XBH"])
        rows.append(
            {
                "slate_date": slate,
                "list_type": list_type,
                "rank": rank,
                "matchup": matchup_label(r),
                "batter_mlbam_id": bid,
                "pitcher_mlbam_id": int(r["pitcher_mlbam_id"]),
                "adj_p_hr": float(r.get("adj_p_hr") or 0),
                "p_hr": float(r.get("p_hr") or 0),
                "adj_p_hit": float(r.get("adj_p_hit") or 0),
                "p_hit": float(r.get("p_hit") or 0),
                "adj_p_xbh": float(r.get("adj_p_xbh") or 0),
                "p_xbh": float(r.get("p_xbh") or 0),
                "PA": pa,
                "H": h,
                "HR": hr,
                "XBH": xbh,
                "hit_ok": h >= 1,
                "hr_ok": hr >= 1,
                "xbh_ok": xbh >= 1,
            }
        )
    return rows


def grand_kpi(rows: list[dict], key: str) -> tuple[int, int, float]:
    y = len(rows)
    x = sum(1 for r in rows if r.get(key))
    pct = (100.0 * x / y) if y else 0.0
    return x, y, pct


def daily_rates(all_by_type: dict[str, list[dict]]) -> list[dict]:
    out = []
    dates = sorted(
        {r["slate_date"] for r in all_by_type["hit"]}
        | {r["slate_date"] for r in all_by_type["hr"]}
        | {r["slate_date"] for r in all_by_type["xbh"]}
    )
    for d in dates:
        def pct_for(lt: str, key: str) -> float:
            chunk = [r for r in all_by_type[lt] if r["slate_date"] == d]
            if not chunk:
                return float("nan")
            return 100.0 * sum(1 for r in chunk if r[key]) / len(chunk)

        out.append(
            {
                "slate_date": d,
                "hit_pct": pct_for("hit", "hit_ok"),
                "xbh_pct": pct_for("xbh", "xbh_ok"),
                "hr_pct": pct_for("hr", "hr_ok"),
            }
        )
    return out


def write_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: ("1" if isinstance(r.get(k), bool) and r[k] else ("0" if isinstance(r.get(k), bool) else r.get(k))) for k in keys})


def write_md(rows_hr: list[dict], rows_hit: list[dict], rows_xbh: list[dict], kpis: dict, path: Path) -> None:
    lines = [
        "# Top-10 prediction audit",
        "",
        "Sources: archived `todays_matchup_predictions.json` under `data/reports/archive/`, outcomes from `statcast_pa_level_league.parquet` (all PAs that calendar day for the batter).",
        "",
        "## All-time KPIs (top-10 lists)",
        "",
        f"- **Hit list:** {kpis['hit'][0]}/{kpis['hit'][1]} — **{kpis['hit'][2]:.1f}%** (≥1 hit)",
        f"- **XBH list:** {kpis['xbh'][0]}/{kpis['xbh'][1]} — **{kpis['xbh'][2]:.1f}%** (≥1 XBH)",
        f"- **HR list:** {kpis['hr'][0]}/{kpis['hr'][1]} — **{kpis['hr'][2]:.1f}%** (≥1 HR)",
        "",
    ]
    for title, rows, focus in [
        ("## HR top-10 rows", rows_hr, "hr"),
        ("## Hit top-10 rows", rows_hit, "hit"),
        ("## XBH top-10 rows", rows_xbh, "xbh"),
    ]:
        lines.append(title)
        lines.append("")
        if not rows:
            lines.append("_No rows._")
            lines.append("")
            continue
        lines.append("| Slate | Rank | Matchup | adj P | raw P | PA | H | HR | XBH | Hit✓ | HR✓ | XBH✓ |")
        lines.append("|---:|---:|---|---:|---:|---:|---:|---:|---:|---|---|---|")
        for r in rows:
            adj = r["adj_p_hr"] if focus == "hr" else r["adj_p_hit"] if focus == "hit" else r["adj_p_xbh"]
            raw = r["p_hr"] if focus == "hr" else r["p_hit"] if focus == "hit" else r["p_xbh"]
            lines.append(
                f"| {r['slate_date']} | {r['rank']} | {r['matchup']} | {adj*100:.2f}% | {raw*100:.2f}% | "
                f"{r['PA']} | {r['H']} | {r['HR']} | {r['XBH']} | "
                f"{'✓' if r['hit_ok'] else '—'} | {'✓' if r['hr_ok'] else '—'} | {'✓' if r['xbh_ok'] else '—'} |"
            )
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def col_minmax(rows: list[dict], cols: list[str]) -> dict[str, tuple[float, float]]:
    mm = {}
    for c in cols:
        vals = [float(r[c]) for r in rows if r.get(c) is not None and not (isinstance(r[c], float) and math.isnan(r[c]))]
        if not vals:
            mm[c] = (0.0, 1.0)
        else:
            mm[c] = (min(vals), max(vals))
    return mm


def table_html(title: str, rows: list[dict], focus: str) -> str:
    if not rows:
        return f"<section><h2>{title}</h2><p><em>No data.</em></p></section>"
    heat_cols = ["adj_focus", "p_focus", "PA", "H", "HR", "XBH"]
    if focus == "hr":
        fadj, fp = "adj_p_hr", "p_hr"
    elif focus == "hit":
        fadj, fp = "adj_p_hit", "p_hit"
    else:
        fadj, fp = "adj_p_xbh", "p_xbh"
    disp_rows = []
    for r in rows:
        dr = dict(r)
        dr["adj_focus"] = dr[fadj]
        dr["p_focus"] = dr[fp]
        disp_rows.append(dr)
    mm = col_minmax(disp_rows, heat_cols)
    thead = "<tr><th>Slate</th><th>Rk</th><th>Matchup</th><th>Adj P</th><th>Raw P</th><th>PA</th><th>H</th><th>HR</th><th>XBH</th><th>Hit✓</th><th>HR✓</th><th>XBH✓</th></tr>"
    body = []
    for r in disp_rows:
        esc = lambda s: str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        row_html = (
            f'<tr><td>{esc(r["slate_date"])}</td><td>{r["rank"]}</td><td>{esc(r["matchup"])}</td>'
            f'<td style="{heat_style(r["adj_focus"], mm["adj_focus"][0], mm["adj_focus"][1])}">{r["adj_focus"]*100:.2f}%</td>'
            f'<td style="{heat_style(r["p_focus"], mm["p_focus"][0], mm["p_focus"][1])}">{r["p_focus"]*100:.2f}%</td>'
            f'<td style="{heat_style(r["PA"], mm["PA"][0], mm["PA"][1])}">{r["PA"]}</td>'
            f'<td style="{heat_style(r["H"], mm["H"][0], mm["H"][1])}">{r["H"]}</td>'
            f'<td style="{heat_style(r["HR"], mm["HR"][0], mm["HR"][1])}">{r["HR"]}</td>'
            f'<td style="{heat_style(r["XBH"], mm["XBH"][0], mm["XBH"][1])}">{r["XBH"]}</td>'
            f'<td>{"✓" if r["hit_ok"] else "—"}</td>'
            f'<td>{"✓" if r["hr_ok"] else "—"}</td>'
            f'<td>{"✓" if r["xbh_ok"] else "—"}</td></tr>'
        )
        body.append(row_html)
    return (
        f'<section class="tblsec"><h2>{title}</h2><div class="tablewrap"><table><thead>{thead}</thead><tbody>'
        + "".join(body)
        + "</tbody></table></div></section>"
    )


def build_html_page(kpis: dict, daily: list[dict], rows_hr: list, rows_hit: list, rows_xbh: list) -> str:
    chart_json = json.dumps(
        {
            "labels": [d["slate_date"] for d in daily],
            "hit": [round(d["hit_pct"], 2) if d["hit_pct"] == d["hit_pct"] else None for d in daily],
            "xbh": [round(d["xbh_pct"], 2) if d["xbh_pct"] == d["xbh_pct"] else None for d in daily],
            "hr": [round(d["hr_pct"], 2) if d["hr_pct"] == d["hr_pct"] else None for d in daily],
        }
    )

    def card(key: str, label: str, sub: str) -> str:
        x, y, p = kpis[key]
        xy_s = f"{x}/{y}" if y else "0/0"
        pct_s = f"{p:.1f}%" if y else "—"
        return f"""<div class="card"><h3>{label}</h3><p class="xy">{xy_s}</p><p class="pct">{pct_s}</p><p class="sub">{sub}</p></div>"""

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>Top-10 prediction audit</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.1/dist/chart.umd.min.js"></script>
<style>
:root {{ font-family: system-ui, sans-serif; background:#0f172a; color:#e2e8f0; }}
body {{ margin:0; padding:1.5rem; max-width:1200px; margin-inline:auto; }}
h1 {{ font-size:1.35rem; margin-bottom:0.5rem; }}
.note {{ color:#94a3b8; font-size:0.9rem; margin-bottom:1.5rem; }}
.kpis {{ display:grid; grid-template-columns:repeat(3,1fr); gap:1rem; margin-bottom:2rem; }}
.card {{ background:#1e293b; border-radius:12px; padding:1rem 1.25rem; border:1px solid #334155; }}
.card h3 {{ margin:0 0 0.5rem; font-size:1rem; color:#94a3b8; font-weight:600; }}
.xy {{ font-size:1.75rem; font-weight:700; margin:0; }}
.pct {{ font-size:1.25rem; color:#38bdf8; margin:0.25rem 0 0; }}
.sub {{ font-size:0.75rem; color:#64748b; margin:0.5rem 0 0; }}
.charts {{ display:grid; grid-template-columns:1fr; gap:1.5rem; margin-bottom:2rem; }}
.chartbox {{ background:#1e293b; border-radius:12px; padding:1rem; border:1px solid #334155; height:280px; }}
.chartbox h3 {{ margin:0 0 0.5rem; font-size:0.95rem; color:#94a3b8; }}
.tablewrap {{ overflow:auto; border-radius:8px; border:1px solid #334155; }}
table {{ border-collapse:collapse; width:100%; font-size:0.82rem; }}
th, td {{ border:1px solid #334155; padding:0.45rem 0.55rem; text-align:left; }}
th {{ background:#1e293b; color:#94a3b8; position:sticky; top:0; }}
.tblsec {{ margin-bottom:2.5rem; }}
.tblsec h2 {{ font-size:1.05rem; margin-bottom:0.75rem; }}
</style>
</head>
<body>
<h1>Top-10 prediction audit</h1>
<p class="note">KPIs = all-time across archived slates. Charts = % of that day&rsquo;s top-10 picks with outcome. Tables = raw rows; numeric cells use orange&rarr;blue at 50% opacity.</p>
<div class="kpis">
{card("hit", "Hits (Hit list)", "All-time: top-10 by Adj P(Hit), ≥1 hit")}
{card("xbh", "XBH (XBH list)", "All-time: top-10 by Adj P(XBH), ≥1 XBH")}
{card("hr", "HRs (HR list)", "All-time: top-10 by Adj P(HR), ≥1 HR")}
</div>
<div class="charts">
<div class="chartbox"><h3>Hit list &mdash; % with ≥1 hit</h3><canvas id="chHit"></canvas></div>
<div class="chartbox"><h3>XBH list &mdash; % with ≥1 XBH</h3><canvas id="chXbh"></canvas></div>
<div class="chartbox"><h3>HR list &mdash; % with ≥1 HR</h3><canvas id="chHr"></canvas></div>
</div>
{table_html("HR top-10 (all slates)", rows_hr, "hr")}
{table_html("Hit top-10 (all slates)", rows_hit, "hit")}
{table_html("XBH top-10 (all slates)", rows_xbh, "xbh")}
<script type="application/json" id="chartData">{chart_json}</script>
<script>
const data = JSON.parse(document.getElementById('chartData').textContent);
const common = {{
  type: 'bar',
  options: {{
    responsive: true,
    maintainAspectRatio: false,
    plugins: {{ legend: {{ display: false }} }},
    scales: {{
      x: {{ ticks: {{ color: '#94a3b8' }}, grid: {{ color: '#334155' }} }},
      y: {{
        min: 0,
        max: 100,
        ticks: {{ color: '#94a3b8', callback: v => v + '%' }},
        grid: {{ color: '#334155' }}
      }}
    }}
  }}
}};
function barChart(id, label, series) {{
  const ctx = document.getElementById(id);
  new Chart(ctx, {{
    ...common,
    data: {{
      labels: data.labels,
      datasets: [{{ label, data: series, backgroundColor: 'rgba(56, 189, 248, 0.55)' }}]
    }}
  }});
}}
if (data.labels.length) {{
  barChart('chHit', 'Hit %', data.hit);
  barChart('chXbh', 'XBH %', data.xbh);
  barChart('chHr', 'HR %', data.hr);
}}
</script>
</body>
</html>"""


def main() -> int:
    paths = sorted(ARCHIVE_DIR.glob("*/todays_matchup_predictions.json")) if ARCHIVE_DIR.is_dir() else []
    if not paths:
        print("No archived predictions found under data/reports/archive/*/todays_matchup_predictions.json", file=sys.stderr)
        print("Run build_matchup_dashboard once to create a dated snapshot.", file=sys.stderr)

    stats = load_batter_day_stats() if paths else pd.DataFrame()

    rows_hr: list[dict] = []
    rows_hit: list[dict] = []
    rows_xbh: list[dict] = []

    for p in paths:
        slate = slate_date_from_path(p)
        if not slate:
            continue
        with open(p, encoding="utf-8") as f:
            raw = json.load(f)
        if not raw:
            continue
        preds = pd.DataFrame(raw)
        if "batter_mlbam_id" not in preds.columns:
            continue
        preds["batter_mlbam_id"] = preds["batter_mlbam_id"].astype(int)
        preds["pitcher_mlbam_id"] = pd.to_numeric(preds.get("pitcher_mlbam_id"), errors="coerce").fillna(0).astype(int)
        # Prefer embedded slate_date when present (must match folder)
        if "slate_date" in preds.columns and preds["slate_date"].notna().any():
            sd = str(preds["slate_date"].dropna().iloc[0])
            if sd:
                slate = sd[:10]

        rows_hr.extend(collect_rows(stats, preds, slate, "hr", "adj_p_hr"))
        rows_hit.extend(collect_rows(stats, preds, slate, "hit", "adj_p_hit"))
        rows_xbh.extend(collect_rows(stats, preds, slate, "xbh", "adj_p_xbh"))

    all_by_type = {"hr": rows_hr, "hit": rows_hit, "xbh": rows_xbh}
    kpis = {
        "hit": grand_kpi(rows_hit, "hit_ok"),
        "xbh": grand_kpi(rows_xbh, "xbh_ok"),
        "hr": grand_kpi(rows_hr, "hr_ok"),
    }

    all_rows = rows_hr + rows_hit + rows_xbh
    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    write_md(rows_hr, rows_hit, rows_xbh, kpis, OUT_MD)
    write_csv(all_rows, OUT_CSV)

    daily = daily_rates(all_by_type) if any(all_by_type.values()) else []
    html = build_html_page(kpis, daily, rows_hr, rows_hit, rows_xbh)
    OUT_HTML.write_text(html, encoding="utf-8")

    print(f"Wrote {OUT_MD}")
    print(f"Wrote {OUT_CSV}")
    print(f"Wrote {OUT_HTML}")
    print(f"Rows: HR={len(rows_hr)} Hit={len(rows_hit)} XBH={len(rows_xbh)} (from {len(paths)} archive file(s))")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
