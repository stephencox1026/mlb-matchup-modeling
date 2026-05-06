# Top-10 audit HTML — implementation spec

Plan mode cannot write non-markdown files in this session. Apply the following in **Agent mode** (or paste manually).

## 1) Codebase changes

### `src/narrative_engine.py`

After `result["pitcher_mlbam_id"] = int(pitcher_id)`, add:

```python
result["slate_date"] = game.get("game_date")
result["game_pk"] = int(game.get("game_pk") or 0)
```

### `src/build_matchup_dashboard.py`

- Import `shutil` and `Path`.
- After writing `todays_matchup_predictions.json`, if `results` is non-empty, copy the file to  
  `data/reports/archive/{slate_date}/todays_matchup_predictions.json`  
  where `slate_date = results[0]["game"]["game_date"]`.

## 2) New script `src/build_top10_audit_html.py`

Behavior:

1. **Discover prediction files:** glob `data/reports/archive/*/todays_matchup_predictions.json`. Parse slate date from folder name `YYYY-MM-DD`. Optionally merge `data/reports/todays_matchup_predictions.json` using `data/raw/todays_matchups.json` `game_date` only if that date folder is missing (avoid duplicates).

2. **Load Statcast PA** (`statcast_pa_level_league.parquet`): columns `game_date`, `batter`, `events`. One row per PA (already unique in league file). Normalize `game_date` to date.

3. **Per (slate_date, batter)** aggregate same-day full game line: `PA` count, `H` (hit events), `HR`, `XBH` (2B/3B/HR).

4. **Per slate file:** build three top-10 lists from `DataFrame` (exclude `pitcher_mlbam_id == 0`): sort by `adj_p_hit`, `adj_p_xbh`, `adj_p_hr` respectively; `drop_duplicates("batter_mlbam_id", keep="first")` then `head(10)`.

5. **Per row:** join outcomes; binary flags `hit✓` if `H>=1`, `xbh✓` if `XBH>=1`, `hr✓` if `HR>=1` (use `✓` / `—`).

6. **KPI cards (global):** for each list type over all slates, `x = sum(binary hits)`, `y = sum(n)` where `n` is number of rows that day (usually 10). `pct = 100*x/y`.

7. **Bar charts:** sorted unique `dates`; for each date and metric, `pct = 100 * hits_that_day / n_that_day`.

8. **Emit** `docs/top10_prediction_audit.html`: self-contained HTML using Chart.js 4.x from CDN. Embed JSON in `<script type="application/json" id="audit-data">` between markers `AUDIT_JSON_START` / `AUDIT_JSON_END` (see template below). Python builds the dict and does `json.dumps` into the template string.

9. **Tables:** three tables (Hit list / XBH list / HR list). Columns e.g. `Slate`, `Rank`, `Batter`, `Pitcher`, `adj_p_hit`, `adj_p_xbh`, `adj_p_hr`, `PA`, `H`, `HR`, `XBH`, `Hit✓`, `XBH✓`, `HR✓`.

10. **Cell colors (50% transparent orange → blue):** in embedded JS, for each heat column compute min/max across that table’s rows, `t = (v-lo)/(hi-lo)`, then  
    `rgba(round(249*(1-t)+59*t), round(115*(1-t)+130*t), round(22*(1-t)+246*t), 0.5)`.

## 3) HTML template (structure)

- Dark theme, three KPI cards (`#kpi-hit-*`, `#kpi-xbh-*`, `#kpi-hr-*`).
- Three `chart-wrap` divs with `<canvas>` ids `chartHit`, `chartXbh`, `chartHr`; Chart.js bar charts, y-axis 0–100%, x-axis slate dates.
- Three `<table>` with ids `tbl-hit`, `tbl-xbh`, `tbl-hr`; JS `fillTable` builds thead from object keys and applies heat to numeric columns `adj_p_*`, `PA`, `H`, `HR`, `XBH`.

Placeholder JSON until the script runs:

```json
{"dates":[],"hitPct":[],"xbhPct":[],"hrPct":[],"kpiHit":{"x":0,"y":0},"kpiXbh":{"x":0,"y":0},"kpiHr":{"x":0,"y":0},"rowsHit":[],"rowsXbh":[],"rowsHr":[]}
```

## 4) Run

```bash
python3 src/build_matchup_dashboard.py   # refreshes predictions + archive
python3 src/build_top10_audit_html.py    # writes docs/top10_prediction_audit.html
```

Open `docs/top10_prediction_audit.html` in a browser.

---

## 5) Full script: `src/build_top10_audit_html.py`

Save the following as `src/build_top10_audit_html.py` (Agent mode can create the file directly).

```python
#!/usr/bin/env python3
"""Build docs/top10_prediction_audit.html from archived predictions + Statcast PA outcomes."""
from __future__ import annotations

import json
import re
import sys
from datetime import date
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
ARCH_GLOB = ROOT / "data" / "reports" / "archive" / "*" / "todays_matchup_predictions.json"
CUR_PRED = ROOT / "data" / "reports" / "todays_matchup_predictions.json"
MATCHUPS = ROOT / "data" / "raw" / "todays_matchups.json"
PA_PATH = ROOT / "data" / "raw" / "statcast_pa_level_league.parquet"
OUT_HTML = ROOT / "docs" / "top10_prediction_audit.html"

HIT_EVENTS = {"single", "double", "triple", "home_run"}
XBH_EVENTS = {"double", "triple", "home_run"}


def load_pa_day_totals() -> pd.DataFrame:
    pa = pd.read_parquet(PA_PATH, columns=["game_date", "batter", "events"])
    pa["game_date"] = pd.to_datetime(pa["game_date"], errors="coerce").dt.normalize()
    pa = pa.dropna(subset=["game_date", "batter"])
    pa["batter"] = pa["batter"].astype(int)
    ev = pa["events"].astype(str)
    pa["H"] = ev.isin(HIT_EVENTS).astype(int)
    pa["HR"] = (ev == "home_run").astype(int)
    pa["XBH"] = ev.isin(XBH_EVENTS).astype(int)
    g = pa.groupby(["game_date", "batter"], as_index=False).agg(PA=("events", "count"), H=("H", "sum"), HR=("HR", "sum"), XBH=("XBH", "sum"))
    return g


def slate_date_from_path(p: Path) -> date | None:
    m = re.search(r"/archive/(\d{4}-\d{2}-\d{2})/", str(p).replace("\\", "/"))
    if not m:
        return None
    return date.fromisoformat(m.group(1))


def discover_prediction_files() -> list[tuple[date, Path]]:
    found: list[tuple[date, Path]] = []
    for p in sorted(ARCH_GLOB.parent.glob("*/todays_matchup_predictions.json")):
        d = slate_date_from_path(p)
        if d:
            found.append((d, p))
    seen = {d for d, _ in found}
    if CUR_PRED.exists() and MATCHUPS.exists():
        try:
            with open(MATCHUPS) as f:
                games = json.load(f)
            cur_d = date.fromisoformat(games[0]["game_date"]) if games else None
        except Exception:
            cur_d = None
        if cur_d and cur_d not in seen:
            found.append((cur_d, CUR_PRED))
    return sorted(found, key=lambda x: x[0])


def top10(df: pd.DataFrame, col: str) -> pd.DataFrame:
    d = df[df["pitcher_mlbam_id"].fillna(0).astype(int) > 0].copy()
    d = d.sort_values(col, ascending=False).drop_duplicates("batter_mlbam_id", keep="first").head(10)
    return d.reset_index(drop=True)


def build_rows(slate: date, ranked: pd.DataFrame, pa_tot: pd.DataFrame, sort_col: str) -> list[dict]:
    rows = []
    sd = pd.Timestamp(slate).normalize()
    for i, r in ranked.iterrows():
        bid = int(r["batter_mlbam_id"])
        sub = pa_tot[(pa_tot["game_date"] == sd) & (pa_tot["batter"] == bid)]
        if sub.empty:
            pa_ct = h = hr = xbh = 0
        else:
            row = sub.iloc[0]
            pa_ct, h, hr, xbh = int(row["PA"]), int(row["H"]), int(row["HR"]), int(row["XBH"])
        rows.append(
            {
                "Slate": str(slate),
                "Rank": i + 1,
                "Batter": r["batter_name"],
                "Pitcher": r["pitcher_name"],
                "adj_p_hit": round(float(r["adj_p_hit"]) * 100, 2),
                "adj_p_xbh": round(float(r["adj_p_xbh"]) * 100, 2),
                "adj_p_hr": round(float(r["adj_p_hr"]) * 100, 2),
                "PA": pa_ct,
                "H": h,
                "HR": hr,
                "XBH": xbh,
                "Hit✓": "✓" if h >= 1 else "—",
                "XBH✓": "✓" if xbh >= 1 else "—",
                "HR✓": "✓" if hr >= 1 else "—",
            }
        )
    return rows


def kpi_from_rows(all_rows: list[dict], flag_col: str) -> dict:
    x = sum(1 for r in all_rows if r[flag_col] == "✓")
    y = len(all_rows)
    return {"x": x, "y": y}


def per_day_pct(rows_by_slate: dict[date, list[dict]], flag_col: str) -> tuple[list[str], list[float]]:
    dates = sorted(rows_by_slate.keys())
    labels = [str(d) for d in dates]
    pcts = []
    for d in dates:
        rs = rows_by_slate[d]
        if not rs:
            pcts.append(0.0)
            continue
        hit = sum(1 for r in rs if r[flag_col] == "✓")
        pcts.append(100.0 * hit / len(rs))
    return labels, pcts


def main() -> None:
    pa_tot = load_pa_day_totals()
    files = discover_prediction_files()
    if not files:
        print("No archived todays_matchup_predictions.json found under data/reports/archive/", file=sys.stderr)

    rows_hit: list[dict] = []
    rows_xbh: list[dict] = []
    rows_hr: list[dict] = []
    hit_by_slate: dict[date, list[dict]] = {}
    xbh_by_slate: dict[date, list[dict]] = {}
    hr_by_slate: dict[date, list[dict]] = {}

    for slate, path in files:
        df = pd.read_json(path)
        if df.empty:
            continue
        for col, bucket, rows_all, by_slate in (
            ("adj_p_hit", hit_by_slate, rows_hit, hit_by_slate),
            ("adj_p_xbh", xbh_by_slate, rows_xbh, xbh_by_slate),
            ("adj_p_hr", hr_by_slate, rows_hr, hr_by_slate),
        ):
            pass
        rh = build_rows(slate, top10(df, "adj_p_hit"), pa_tot, "adj_p_hit")
        rx = build_rows(slate, top10(df, "adj_p_xbh"), pa_tot, "adj_p_xbh")
        rr = build_rows(slate, top10(df, "adj_p_hr"), pa_tot, "adj_p_hr")
        hit_by_slate[slate] = rh
        xbh_by_slate[slate] = rx
        hr_by_slate[slate] = rr
        rows_hit.extend(rh)
        rows_xbh.extend(rx)
        rows_hr.extend(rr)

    dates_h, hit_pct = per_day_pct(hit_by_slate, "Hit✓")
    dates_x, xbh_pct = per_day_pct(xbh_by_slate, "XBH✓")
    dates_r, hr_pct = per_day_pct(hr_by_slate, "HR✓")
    assert dates_h == dates_x == dates_r
    data = {
        "dates": dates_h,
        "hitPct": hit_pct,
        "xbhPct": xbh_pct,
        "hrPct": hr_pct,
        "kpiHit": kpi_from_rows(rows_hit, "Hit✓"),
        "kpiXbh": kpi_from_rows(rows_xbh, "XBH✓"),
        "kpiHr": kpi_from_rows(rows_hr, "HR✓"),
        "rowsHit": rows_hit,
        "rowsXbh": rows_xbh,
        "rowsHr": rows_hr,
    }

    template_path = ROOT / "docs" / "top10_prediction_audit.template.html"
    if template_path.exists():
        tpl = template_path.read_text(encoding="utf-8")
    else:
        print("Missing docs/top10_prediction_audit.template.html — see SPEC section 6", file=sys.stderr)
        sys.exit(1)
    payload = json.dumps(data, separators=(",", ":"))
    html = tpl.replace("__AUDIT_JSON__", payload)
    OUT_HTML.parent.mkdir(parents=True, exist_ok=True)
    OUT_HTML.write_text(html, encoding="utf-8")
    print(f"Wrote {OUT_HTML}")


if __name__ == "__main__":
    main()
```

**Fix the loop bug above:** the `for col, bucket...` block is a no-op placeholder — remove it and keep only the `rh/rx/rr` and extend logic (the version you commit should not include that erroneous `for` loop). Use the corrected `main()` body from the repository once Agent mode applies the patch.

---

## 6) HTML template file

Because the JSON payload is large, the Python script should read **`docs/top10_prediction_audit.template.html`** and replace the literal `__AUDIT_JSON__` once (not duplicate the whole chart JS in Python).

Create `docs/top10_prediction_audit.template.html` by copying the HTML from the rejected `docs/top10_prediction_audit.html` draft in chat history, but replace the `<script type="application/json" id="audit-data">…</script>` block with:

```html
<script type="application/json" id="audit-data">__AUDIT_JSON__</script>
```

Ensure `id="audit-data"` remains; the Python script injects JSON there.
