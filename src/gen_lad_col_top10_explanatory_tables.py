#!/usr/bin/env python3
"""LAD @ COL: top 10 per team × HR, Hit, XBH — compact columns + Coors park-table lean."""
from __future__ import annotations

import json
from pathlib import Path

from lad_col_park_factor_report import coors_table_adjust, scale_adj

ROOT = Path(__file__).resolve().parents[1]
PRED = ROOT / "data" / "reports" / "todays_matchup_predictions.json"
OUT = ROOT / "data" / "reports" / "lad_col_top10_brief.md"

TEAM_NAMES = {"LAD": "Los Angeles Dodgers", "COL": "Colorado Rockies"}


def safe_cell(s: str, max_len: int = 160) -> str:
    if not s:
        return "—"
    t = " ".join(s.split()).replace("|", "/")
    return (t[: max_len - 1] + "…") if len(t) > max_len else t


def brief_explanation(r: dict, kind: str) -> str:
    parts = []
    if r.get("top_reasons"):
        parts.append("; ".join(r["top_reasons"][:2]))
    hn = r.get("hit_narrative") or ""
    # Sentence boundaries only — do not split on decimals in "5.3%".
    first_sent = hn.split(". ")[0].strip() if hn else ""
    if first_sent and first_sent not in " ".join(parts):
        if not first_sent.endswith("."):
            first_sent += "."
        parts.append(first_sent)
    if kind in ("hr", "xbh") and r.get("bvp_text"):
        parts.append(r["bvp_text"])
    return safe_cell(" ".join(parts))


def grade_for(r: dict, label_key: str, conf_key: str) -> str:
    g = r.get(label_key)
    if g:
        return str(g)
    c = r.get(conf_key)
    if c is None:
        return "—"
    c = float(c)
    if c >= 1.05:
        return "High"
    if c >= 0.88:
        return "Medium"
    if c >= 0.75:
        return "Low"
    return "Very Low"


def rows_for_team(matchups: list[dict], team: str, sort_key: str) -> list[dict]:
    sub = [r for r in matchups if r.get("batter_team") == team]
    return sorted(sub, key=lambda x: x[sort_key], reverse=True)[:10]


def park_adj_bundle(r: dict) -> dict[str, float]:
    """Park-table adjusted Adj P for hit / hr / xbh (Coors row + carry damping)."""
    ph, phr, px = coors_table_adjust(r["p_hit"], r["p_hr"], r["p_xbh"])
    return {
        "hit": scale_adj(float(r["adj_p_hit"]), float(r["p_hit"]), ph),
        "hr": scale_adj(float(r["adj_p_hr"]), float(r["p_hr"]), phr),
        "xbh": scale_adj(float(r["adj_p_xbh"]), float(r["p_xbh"]), px),
    }


def suggestion_today(metric: str, r: dict, padj: dict, label_key: str, conf_key: str) -> str:
    """Model + Coors table (HR/1B/2B3B) with carry damping from the graphic key."""
    adj = float(r[{"hr": "adj_p_hr", "hit": "adj_p_hit", "xbh": "adj_p_xbh"}[metric]])
    wadj = padj[metric]
    delta = 100.0 * (wadj - adj) / adj if adj > 1e-8 else 0.0
    grade = grade_for(r, label_key, conf_key)
    tier = r.get("tier") or "Average"

    env = "Coors table + carry damp (cool, dry, mixed wind)."

    if metric == "hr":
        if grade in ("Medium", "High"):
            if wadj >= 0.04:
                core = "Best HR cred on this list—still a dart, but park table backs the model."
            elif wadj >= 0.028:
                core = "Top HR confidence in file; raw HR line still modest—keep stakes small."
            else:
                core = "HR confidence > raw HR—sprinkle only, let park row widen misses."
        elif grade == "Low" and wadj >= 0.04:
            core = "HR in play for GPP; model confidence only middling."
        elif wadj >= 0.034 and grade != "Very Low":
            core = "Secondary HR look; thin-air bump is real, trust is mixed."
        elif wadj >= 0.034:
            core = "Fly-ball noise possible, but Very Low confidence—tiny HR stake only."
        elif wadj >= 0.022:
            core = "Prefer hit/singles; HR needs a mistake pitch."
        else:
            core = "Skip HR; park table cannot overcome this matchup line."
        tail = f"Park blend ~{delta:+.0f}% vs model HR (same carry math for each batter)."
    elif metric == "hit":
        if wadj >= 0.24:
            core = "Lean over on getting on base via hit."
        elif wadj >= 0.18:
            core = "Solid single-knock profile tonight."
        elif wadj >= 0.14:
            core = "Coin-flip hit; punchies still matter."
        else:
            core = "Underweight hits; park table cannot fix contact risk."
        tail = f"Park-blend hit line ~{delta:+.1f}% vs model."
    else:
        if wadj >= 0.095:
            core = "XBH (2B/3B/HR) is the cleanest power route—park loves non-HR XBHs."
        elif wadj >= 0.065:
            core = "XBH reasonable; doubles pick up most of the stadium +33% leg."
        else:
            core = "Singles/speed more likely than extra bases."
        tail = f"Park-blend XBH line ~{delta:+.1f}% vs model."

    conf_bit = f"{tier} / {grade} conf."
    return safe_cell(f"{core} {tail} {env} {conf_bit}", max_len=260)


def build_table(
    m: list[dict],
    title: str,
    metric: str,
    sort_key: str,
    label_key: str,
    conf_key: str,
) -> list[str]:

    lines = [
        f"## {title}",
    ]
    if title == "Home runs":
        lines.append(
            "*HR park note: Ballpark Pal HR (+11%) scales every batter’s HR leg the same way after carry damping, "
            "so the “% vs model” is nearly identical row-to-row; read differences off **grade** and **Adj P** rank.*"
        )
    lines += [
        "",
        "| Team | Player | Pitcher facing | Tier | Grade | Brief explanation | Today's lean (model + Coors park blend) |",
        "|:---|:---|:---|:---|:---|:---|:---|",
    ]

    for team in ("LAD", "COL"):
        for r in rows_for_team(m, team, sort_key):
            team_name = TEAM_NAMES.get(team, team)
            throws = "LHP" if r.get("pitcher_throws") == "L" else "RHP"
            pitcher = f"{r['pitcher_name']} ({throws})"
            pa = park_adj_bundle(r)
            sug = suggestion_today(metric, r, pa, label_key, conf_key)
            lines.append(
                f"| {team_name} | {r['batter_name']} | {pitcher} | {r.get('tier', '—')} | "
                f"{grade_for(r, label_key, conf_key)} | {brief_explanation(r, metric)} | {sug} |"
            )
    lines.append("")
    return lines


def main() -> None:
    with open(PRED) as f:
        all_rows = json.load(f)
    m = [
        r
        for r in all_rows
        if (r.get("batter_team") == "LAD" and r.get("pitcher_team") == "COL")
        or (r.get("batter_team") == "COL" and r.get("pitcher_team") == "LAD")
    ]

    blocks = [
        "# LAD @ Colorado — Top 10 per team (brief)",
        "",
        f"**Source:** `{PRED.relative_to(ROOT)}`. Order = model **Adj P** for that outcome.",
        "",
        "**Today's lean** blends the model with the Ballpark Pal Coors row (HR +11%, 1B +11%, 2B/3B +33%) "
        "and carry damping from your graphic key: cold upper-50s, red H (dry), mixed wind, "
        "evening pitcher-friendly — see `src/lad_col_park_factor_report.py`.",
        "",
    ]
    blocks.extend(
        build_table(m, "Home runs", "hr", "adj_p_hr", "conf_hr_label", "conf_hr")
    )
    blocks.extend(
        build_table(m, "Hits", "hit", "adj_p_hit", "conf_hit_label", "conf_hit")
    )
    blocks.extend(
        build_table(m, "XBH", "xbh", "adj_p_xbh", "conf_xbh_label", "conf_xbh")
    )

    OUT.write_text("\n".join(blocks), encoding="utf-8")
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
