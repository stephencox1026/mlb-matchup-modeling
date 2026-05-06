#!/usr/bin/env python3
"""Emit Sections 1–6 markdown tables per daily-output-sections.mdc rules."""
import json
import os
import sys

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
sys.path.insert(0, os.path.join(ROOT, "src"))
from narrative_engine import confidence_label_for_target  # noqa: E402


def confidence_label(conf):
    if conf >= 1.05:
        return "High"
    if conf >= 0.88:
        return "Medium"
    if conf >= 0.75:
        return "Low"
    return "Very Low"


def load_pa_level():
    pa = pd.read_parquet("data/raw/statcast_pa_level_league.parquet")
    hit_events = {"single", "double", "triple", "home_run"}
    pa["is_hit"] = pa["events"].isin(hit_events).astype(int)
    pa["is_hr"] = (pa["events"] == "home_run").astype(int)
    pa["is_xbh"] = pa["events"].isin({"double", "triple", "home_run"}).astype(int)
    pa["is_k"] = pa["events"].str.contains("strikeout", case=False, na=False).astype(int)
    if "pitch_number" in pa.columns:
        pa = pa.sort_values(["game_pk", "at_bat_number", "batter", "pitch_number"])
    return pa.groupby(["game_pk", "at_bat_number", "batter"], as_index=False).tail(1)


def career_table(last_pa):
    g = last_pa.groupby("batter").agg(
        pa=("events", "count"),
        hr=("is_hr", "sum"),
        hit=("is_hit", "sum"),
        xbh=("is_xbh", "sum"),
    )
    g["hr_rate"] = g["hr"] / g["pa"]
    g["hit_rate"] = g["hit"] / g["pa"]
    g["xbh_rate"] = g["xbh"] / g["pa"]
    return g


def bvp_cell_from_counts(h, n, k):
    if n == 0:
        return "No BvP history"
    ba = h / n
    ba_s = f"{ba:.3f}"
    if ba_s.startswith("0."):
        ba_disp = "(" + ba_s[1:] + ")"
    else:
        ba_disp = "(" + ba_s + ")"
    return f"{int(h)}-for-{int(n)} {ba_disp}, {int(k)} K"


def three_pa(raw_p):
    return (1 - (1 - raw_p) ** 3) * 100


def matchup_label(row):
    return f"{row['batter_name']} ({row['batter_team']}) vs {row['pitcher_name']}"


def attach_bvp(df, bvp_agg):
    out = df.merge(
        bvp_agg,
        how="left",
        left_on=["batter_mlbam_id", "pitcher_mlbam_id"],
        right_on=["batter", "pitcher"],
    )
    for c in ["bvp_n", "bvp_h", "bvp_k", "bvp_hr", "bvp_xbh"]:
        if c in out.columns:
            out[c] = out[c].fillna(0)
    return out


def main():
    pred_path = os.environ.get(
        "MATCHUP_PREDICTIONS_JSON", "data/reports/todays_matchup_predictions.json"
    )
    preds = json.load(open(pred_path))
    df = pd.DataFrame(preds)
    last_pa = load_pa_level()
    career = career_table(last_pa)

    bvp_agg = last_pa.groupby(["batter", "pitcher"], as_index=False).agg(
        bvp_n=("events", "count"),
        bvp_h=("is_hit", "sum"),
        bvp_k=("is_k", "sum"),
        bvp_hr=("is_hr", "sum"),
        bvp_xbh=("is_xbh", "sum"),
    )
    dfx = attach_bvp(df, bvp_agg)

    def bvp_cell(row):
        n = int(row["bvp_n"]) if "bvp_n" in row.index else 0
        if n <= 0:
            return "No BvP history"
        return bvp_cell_from_counts(row["bvp_h"], n, row["bvp_k"])

    def score_col(row, target):
        """Prefer new score_* fields, fall back to deprecated adj_p_* alias."""
        return row.get(f"score_{target}", row.get(f"adj_p_{target}", 0.0))

    def cal_p_col(row, target):
        """Prefer new p_*_calibrated, fall back to raw p_*."""
        return row.get(f"p_{target}_calibrated", row.get(f"p_{target}", 0.0))

    # Rank by raw per-PA probability (post posterior blend); score_* remains a column only.
    sort_hr = "p_hr" if "p_hr" in dfx.columns else ("score_hr" if "score_hr" in dfx.columns else "adj_p_hr")
    sort_hit = "p_hit" if "p_hit" in dfx.columns else ("score_hit" if "score_hit" in dfx.columns else "adj_p_hit")
    sort_xbh = "p_xbh" if "p_xbh" in dfx.columns else ("score_xbh" if "score_xbh" in dfx.columns else "adj_p_xbh")

    # Section 1
    s1 = dfx.sort_values(sort_hr, ascending=False).head(25)
    print("## Section 1: Top 25 — Home Runs")
    print()
    print(
        "| Rank | Batter vs Pitcher | Score | Cal P(HR) | Raw P(HR) | 3-PA | Conf | Confidence | Career HR Rate | BvP |"
    )
    print("|---:|---|---:|---:|---:|---:|---:|---|---|---|")
    for i, (_, r) in enumerate(s1.iterrows(), 1):
        bid = int(r["batter_mlbam_id"]) if pd.notna(r["batter_mlbam_id"]) else None
        if bid and bid in career.index:
            c = career.loc[bid]
            career_s = f"{c['hr_rate']*100:.1f}% ({int(c['hr'])} HR)"
        else:
            career_s = "—"
        print(
            f"| {i} | {matchup_label(r)} | {score_col(r,'hr')*100:.2f} | "
            f"{cal_p_col(r,'hr')*100:.1f}% | {r['p_hr']*100:.1f}% | "
            f"{three_pa(r['p_hr']):.1f}% | {r['conf_hr']:.3f} | {r['conf_hr_label']} | {career_s} | {bvp_cell(r)} |"
        )
    print()

    s2 = dfx.sort_values(sort_hit, ascending=False).head(25)
    print("## Section 2: Top 25 — Hits")
    print()
    print(
        "| Rank | Batter vs Pitcher | Score | Cal P(Hit) | Raw P(Hit) | 3-PA | Conf | Confidence | Career BA | BvP |"
    )
    print("|---:|---|---:|---:|---:|---:|---:|---|---|---|")
    for i, (_, r) in enumerate(s2.iterrows(), 1):
        bid = int(r["batter_mlbam_id"]) if pd.notna(r["batter_mlbam_id"]) else None
        if bid and bid in career.index:
            c = career.loc[bid]
            career_s = f".{int(c['hit_rate']*1000):03d}"
        else:
            career_s = "—"
        print(
            f"| {i} | {matchup_label(r)} | {score_col(r,'hit')*100:.2f} | "
            f"{cal_p_col(r,'hit')*100:.1f}% | {r['p_hit']*100:.1f}% | "
            f"{three_pa(r['p_hit']):.1f}% | {r['conf_hit']:.3f} | {r['conf_hit_label']} | {career_s} | {bvp_cell(r)} |"
        )
    print()

    s3 = dfx.sort_values(sort_xbh, ascending=False).head(25)
    print("## Section 3: Top 25 — Extra-Base Hits (XBH)")
    print()
    print(
        "| Rank | Batter vs Pitcher | Score | Cal P(XBH) | Raw P(XBH) | 3-PA | Conf | Confidence | Career XBH Rate | BvP | BvP XBH |"
    )
    print("|---:|---|---:|---:|---:|---:|---:|---|---|---|---:|")
    for i, (_, r) in enumerate(s3.iterrows(), 1):
        bid = int(r["batter_mlbam_id"]) if pd.notna(r["batter_mlbam_id"]) else None
        if bid and bid in career.index:
            c = career.loc[bid]
            career_s = f"{c['xbh_rate']*100:.1f}%"
        else:
            career_s = "—"
        xbh_grade = (
            r.get("conf_xbh_label")
            or confidence_label_for_target(float(r["conf_xbh"]), "xbh")
        )
        bvp_n = int(r["bvp_n"])
        bvp_xbh_disp = "—" if bvp_n <= 0 else int(r["bvp_xbh"])
        print(
            f"| {i} | {matchup_label(r)} | {score_col(r,'xbh')*100:.2f} | "
            f"{cal_p_col(r,'xbh')*100:.1f}% | {r['p_xbh']*100:.1f}% | "
            f"{three_pa(r['p_xbh']):.1f}% | {r['conf_xbh']:.3f} | {xbh_grade} | {career_s} | {bvp_cell(r)} | {bvp_xbh_disp} |"
        )
    print()

    s4 = dfx[dfx["bvp_hr"] > 0].sort_values(["bvp_hr", sort_hr], ascending=[False, False]).head(25)
    print("## Section 4: BvP-Favorable — Home Runs")
    print()
    print(
        "| Rank | Batter vs Pitcher | Score | Cal P(HR) | Raw P(HR) | 3-PA | Conf | Confidence | Career HR Rate | BvP | BvP HR |"
    )
    print("|---:|---|---:|---:|---:|---:|---:|---|---|---|---:|")
    for i, (_, r) in enumerate(s4.iterrows(), 1):
        bid = int(r["batter_mlbam_id"]) if pd.notna(r["batter_mlbam_id"]) else None
        if bid and bid in career.index:
            c = career.loc[bid]
            career_s = f"{c['hr_rate']*100:.1f}% ({int(c['hr'])} HR)"
        else:
            career_s = "—"
        print(
            f"| {i} | {matchup_label(r)} | {score_col(r,'hr')*100:.2f} | "
            f"{cal_p_col(r,'hr')*100:.1f}% | {r['p_hr']*100:.1f}% | "
            f"{three_pa(r['p_hr']):.1f}% | {r['conf_hr']:.3f} | {r['conf_hr_label']} | {career_s} | {bvp_cell(r)} | {int(r['bvp_hr'])} |"
        )
    print()

    s5 = dfx[(dfx["bvp_n"] >= 5) & (dfx["bvp_h"] / dfx["bvp_n"] >= 0.250)].sort_values(
        ["bvp_h", sort_hit], ascending=[False, False]
    ).head(25)
    print("## Section 5: BvP-Favorable — Hits")
    print()
    print(
        "| Rank | Batter vs Pitcher | Score | Cal P(Hit) | Raw P(Hit) | 3-PA | Conf | Confidence | Career BA | BvP | BvP Hits |"
    )
    print("|---:|---|---:|---:|---:|---:|---:|---|---|---|---:|")
    for i, (_, r) in enumerate(s5.iterrows(), 1):
        bid = int(r["batter_mlbam_id"]) if pd.notna(r["batter_mlbam_id"]) else None
        if bid and bid in career.index:
            c = career.loc[bid]
            career_s = f".{int(c['hit_rate']*1000):03d}"
        else:
            career_s = "—"
        print(
            f"| {i} | {matchup_label(r)} | {score_col(r,'hit')*100:.2f} | "
            f"{cal_p_col(r,'hit')*100:.1f}% | {r['p_hit']*100:.1f}% | "
            f"{three_pa(r['p_hit']):.1f}% | {r['conf_hit']:.3f} | {r['conf_hit_label']} | {career_s} | {bvp_cell(r)} | {int(r['bvp_h'])} |"
        )
    print()

    s6 = dfx[dfx["bvp_xbh"] > 0].sort_values(["bvp_xbh", sort_xbh], ascending=[False, False]).head(25)
    print("## Section 6: BvP-Favorable — XBH")
    print()
    print(
        "| Rank | Batter vs Pitcher | Score | Cal P(XBH) | Raw P(XBH) | 3-PA | Conf | Confidence | Career XBH Rate | BvP | BvP XBH |"
    )
    print("|---:|---|---:|---:|---:|---:|---:|---|---|---|---:|")
    for i, (_, r) in enumerate(s6.iterrows(), 1):
        bid = int(r["batter_mlbam_id"]) if pd.notna(r["batter_mlbam_id"]) else None
        if bid and bid in career.index:
            c = career.loc[bid]
            career_s = f"{c['xbh_rate']*100:.1f}%"
        else:
            career_s = "—"
        xbh_grade = (
            r.get("conf_xbh_label")
            or confidence_label_for_target(float(r["conf_xbh"]), "xbh")
        )
        print(
            f"| {i} | {matchup_label(r)} | {score_col(r,'xbh')*100:.2f} | "
            f"{cal_p_col(r,'xbh')*100:.1f}% | {r['p_xbh']*100:.1f}% | "
            f"{three_pa(r['p_xbh']):.1f}% | {r['conf_xbh']:.3f} | {xbh_grade} | {career_s} | {bvp_cell(r)} | {int(r['bvp_xbh'])} |"
        )


if __name__ == "__main__":
    main()
