#!/usr/bin/env python3
"""LAD @ COL: top-10 HR / hit / XBH from model JSON + Ballpark Pal Coors row (park table).

Stadium table (Ballpark Pal Daily Stadium, Coors): HR +11%, 2B/3B +33%, 1B +11%.
Graphic key + narrative: cold (upper 50s), red H (dry, worse carry), mixed wind,
‘pitcher-friendly’ evening → dampen carry on those boosts via CARRY_DAMP on the
increment above 1.0 for each stadium K.
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PRED = ROOT / "data" / "reports" / "todays_matchup_predictions.json"
OUT = ROOT / "data" / "reports" / "lad_col_2026-04-18_model_park.md"

K_HR_ROW = 1.11
K_23_ROW = 1.33
K_1B_ROW = 1.11
CARRY_DAMP = 0.97 * 0.96 * 0.98 * 0.97


def eff_mult(row_k: float) -> float:
    return 1.0 + (row_k - 1.0) * CARRY_DAMP


M_HR = eff_mult(K_HR_ROW)
M_23 = eff_mult(K_23_ROW)
M_1B = eff_mult(K_1B_ROW)


def coors_table_adjust(p_hit: float, p_hr: float, p_xbh: float) -> tuple[float, float, float]:
    p_hr = max(0.0, min(float(p_hr), float(p_xbh), float(p_hit)))
    p_hit = float(p_hit)
    p_xbh = float(p_xbh)
    single = max(0.0, p_hit - p_xbh)
    non_hr_xbh = max(0.0, p_xbh - p_hr)

    p_hr_n = min(0.48, p_hr * M_HR)
    non_n = min(0.45, non_hr_xbh * M_23)
    single_n = min(0.45, single * M_1B)

    p_xbh_n = min(0.85, p_hr_n + non_n)
    p_hit_n = min(0.58, single_n + p_xbh_n)
    return p_hit_n, p_hr_n, p_xbh_n


def scale_adj(adj: float, base: float, new: float) -> float:
    if base <= 1e-9:
        return adj
    return round(adj * (new / base), 5)


def confidence_label(conf: float) -> str:
    if conf >= 1.05:
        return "High"
    if conf >= 0.88:
        return "Medium"
    if conf >= 0.75:
        return "Low"
    return "Very Low"


def fmt_row(i: int, r: dict, rawk: str, adjk: str, wxk: str, wadj: str, confk: str, gradelk: str) -> str:
    grade = r.get(gradelk) or confidence_label(float(r.get(confk, 0)))
    return (
        f"| {i} | {r['batter_name']} | {r['pitcher_name']} ({r['pitcher_throws']}HP) "
        f"| {r[rawk]:.4f} | {r[adjk]:.4f} | {r[wxk]:.4f} | {r[wadj]:.4f} "
        f"| {r[confk]} | {grade} |"
    )


def main() -> None:
    with open(PRED) as f:
        rows = json.load(f)

    lad_col = [
        r
        for r in rows
        if (r.get("batter_team") == "LAD" and r.get("pitcher_team") == "COL")
        or (r.get("batter_team") == "COL" and r.get("pitcher_team") == "LAD")
    ]

    for r in lad_col:
        ph, phr, px = coors_table_adjust(r["p_hit"], r["p_hr"], r["p_xbh"])
        r["_w_p_hit"] = ph
        r["_w_p_hr"] = phr
        r["_w_p_xbh"] = px
        r["_w_adj_p_hit"] = scale_adj(r["adj_p_hit"], r["p_hit"], ph)
        r["_w_adj_p_hr"] = scale_adj(r["adj_p_hr"], r["p_hr"], phr)
        r["_w_adj_p_xbh"] = scale_adj(r["adj_p_xbh"], r["p_xbh"], px)

    lines: list[str] = [
        "# LAD @ COL — model top 10 (both teams) + Coors park-table adjustments",
        "",
        f"**Source:** `{PRED.relative_to(ROOT)}`.",
        "",
        "## Park-table methodology",
        "",
        "**Stadium table (Coors, LAD@COL):** HR +11%, 2B/3B +33%, singles +11%.",
        "Decompose `P(hit)` into singles `P(hit)−P(XBH)`, non-HR XBH `P(XBH)−P(HR)`, and HR `P(HR)`; apply row-specific K; cap and re-sum.",
        "",
        "**Graphic key + slate note:** upper-50s (cold band vs carry), **red H** (dry → worse carry), mixed wind, Ballpark Pal ‘pitcher-friendly evening’.",
        f"**CARRY_DAMP** = {CARRY_DAMP:.4f} multiplies the increment `(K−1)` for each stadium factor → effective **M_HR={M_HR:.4f}**, **M_2B3B={M_23:.4f}**, **M_1B={M_1B:.4f}**.",
        "",
        "`Adj P` after park table = model `Adj P` × (park row raw / model raw) so confidence tier is unchanged.",
        "",
    ]

    blocks = [
        ("Home run — rank by model Adj P(HR)", "adj_p_hr", "p_hr", "adj_p_hr", "_w_p_hr", "_w_adj_p_hr", "conf_hr", "conf_hr_label"),
        ("Home run — rank by park-table Adj P(HR)", "_w_adj_p_hr", "p_hr", "adj_p_hr", "_w_p_hr", "_w_adj_p_hr", "conf_hr", "conf_hr_label"),
        ("Hit — rank by model Adj P(Hit)", "adj_p_hit", "p_hit", "adj_p_hit", "_w_p_hit", "_w_adj_p_hit", "conf_hit", "conf_hit_label"),
        ("Hit — rank by park-table Adj P(Hit)", "_w_adj_p_hit", "p_hit", "adj_p_hit", "_w_p_hit", "_w_adj_p_hit", "conf_hit", "conf_hit_label"),
        ("XBH — rank by model Adj P(XBH)", "adj_p_xbh", "p_xbh", "adj_p_xbh", "_w_p_xbh", "_w_adj_p_xbh", "conf_xbh", "conf_xbh_label"),
        ("XBH — rank by park-table Adj P(XBH)", "_w_adj_p_xbh", "p_xbh", "adj_p_xbh", "_w_p_xbh", "_w_adj_p_xbh", "conf_xbh", "conf_xbh_label"),
    ]

    for team in ("LAD", "COL"):
        sub = [r for r in lad_col if r.get("batter_team") == team]
        lines.append(f"## {team} batters")
        lines.append("")
        for title, sort_key, rawk, adjk, wxk, wadj, confk, gradelk in blocks:
            ranked = sorted(sub, key=lambda r: r[sort_key], reverse=True)[:10]
            lines.append(f"### {title}")
            lines.append("")
            lines.append(
                "| # | Batter | vs SP | Raw P | Adj P (model) | P (park) | Adj P (park) | Conf | Grade |"
            )
            lines.append("|---:|---|---:|---:|---:|---:|---:|---:|---|")
            for i, r in enumerate(ranked, 1):
                lines.append(fmt_row(i, r, rawk, adjk, wxk, wadj, confk, gradelk))
            lines.append("")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {OUT} ({len(lad_col)} matchups)")


if __name__ == "__main__":
    main()
