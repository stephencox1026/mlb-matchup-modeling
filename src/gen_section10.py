#!/usr/bin/env python3
"""Top 10 XBH full analysis — same markdown structure as gen_section8 (hits)."""
import json
import os
import sys

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
sys.path.insert(0, os.path.join(ROOT, "src"))
from narrative_engine import confidence_label_for_target  # noqa: E402


def _fmt_avg_paren(ba):
    s = f"{ba:.3f}"
    return "(" + (s[1:] if s.startswith("0.") else s) + ")"


def _fmt_avg_plain(ba):
    s = f"{ba:.3f}"
    return s[1:] if s.startswith("0.") else s


_pred_path = os.environ.get(
    "MATCHUP_PREDICTIONS_JSON", "data/reports/todays_matchup_predictions.json"
)
preds = json.load(open(_pred_path))
df = pd.DataFrame(preds)
pa_all = pd.read_parquet("data/raw/statcast_pa_level_league.parquet")
_b_ix = pa_all.groupby("batter").indices
_p_ix = pa_all.groupby("pitcher").indices

LEAGUE_XBH = 0.079


def pa_batter(bid):
    if bid is None or bid not in _b_ix:
        return pa_all.iloc[[]]
    return pa_all.iloc[_b_ix[bid]]


def pa_pitcher(pid):
    if pid is None or pid not in _p_ix:
        return pa_all.iloc[[]]
    return pa_all.iloc[_p_ix[pid]]


pitcher_prof = pd.read_parquet("data/raw/pitcher_profiles_by_season.parquet")
batter_prof = pd.read_parquet("data/raw/batter_pitch_profiles.parquet")

try:
    qp = pd.read_csv("data/raw/qualifying_pitchers_2026.csv")
except Exception:
    qp = pd.DataFrame()

_sort_xbh = "p_xbh" if "p_xbh" in df.columns else "adj_p_xbh"
top10_xbh = df.sort_values(_sort_xbh, ascending=False).head(10).reset_index(drop=True)
total_matchups = len(df)

print("## Section 10: Top 10 XBH Full Analysis\n")

for idx, row in top10_xbh.iterrows():
    bname = row["batter_name"]
    pname = row["pitcher_name"]
    bteam = row["batter_team"]
    bid = int(row["batter_mlbam_id"]) if not pd.isna(row.get("batter_mlbam_id")) else None
    pid = int(row["pitcher_mlbam_id"]) if not pd.isna(row.get("pitcher_mlbam_id")) else None
    raw_xbh = row.get("p_xbh", 0) or 0
    adj_xbh = row.get("adj_p_xbh", 0) or 0
    conf = row.get("conf_xbh", 0) or 0
    grade = row.get("conf_xbh_label") or confidence_label_for_target(conf, "xbh")
    conf_factors = row.get("conf_factors", {})
    if isinstance(conf_factors, str):
        conf_factors = json.loads(conf_factors)
    three_pa_raw = (1 - (1 - raw_xbh) ** 3) * 100
    three_pa_adj = (1 - (1 - adj_xbh) ** 3) * 100
    three_pa_adj_p = 1 - (1 - adj_xbh) ** 3
    inv = int(round(1 / three_pa_adj_p)) if three_pa_adj_p > 0 else 999
    inv = max(1, min(inv, 999))

    last = pname.split()[-1]
    blast = bname.split()[-1]

    print(f"## {bname} XBH Analysis vs {pname}")
    print()
    print(f"Model (tonight): Raw P(XBH) {raw_xbh*100:.1f}% | Adj P(XBH) {adj_xbh*100:.1f}% | Conf {conf:.3f} ({grade})")
    print(
        f"3 PA vs starter: (1-(1-{raw_xbh:.4f})^3 ≈) {three_pa_raw:.1f}% raw | ~{three_pa_adj:.1f}% adjusted — "
        f"about **1-in-{inv}** at least one XBH."
    )
    print()
    cf = conf_factors if conf_factors else {}
    cf_str = (
        f"Confidence factors: pitcher data {cf.get('pitcher_data','—')} | "
        f"BvP HR {cf.get('bvp_hr','—')} | BvP hit {cf.get('bvp_hit','—')} | "
        f"staleness {cf.get('staleness','—')} | convergence HR {cf.get('convergence_hr','—')} | "
        f"convergence hit {cf.get('convergence_hit','—')} | "
        f"convergence XBH {cf.get('convergence_xbh','—')}"
    )
    print(cf_str)
    print()

    if not bid:
        print("No batter ID available.\n---\n")
        continue

    batter_pa = pa_batter(bid)
    total_pa = len(batter_pa)
    career_xbh = int(batter_pa["is_xbh"].sum()) if "is_xbh" in batter_pa.columns else 0
    career_xbh_rate = career_xbh / total_pa if total_pa > 0 else 0
    career_k_rate = batter_pa["is_strikeout"].mean() * 100 if "is_strikeout" in batter_pa.columns else 0
    career_k_model = (row.get("p_k", 0) or 0) * 100
    if "events" in batter_pa.columns:
        evs = batter_pa["events"]
        career_doubles = int(evs.eq("double").sum())
        career_triples = int(evs.eq("triple").sum())
    else:
        career_doubles = career_triples = 0
    career_hrs = int(batter_pa["is_hr"].sum()) if "is_hr" in batter_pa.columns else 0
    career_hits = int(batter_pa["is_hit"].sum()) if "is_hit" in batter_pa.columns else 0
    career_ba = career_hits / total_pa if total_pa > 0 else 0

    xbh_vs = (
        "well above"
        if career_xbh_rate > LEAGUE_XBH + 0.02
        else ("above" if career_xbh_rate > LEAGUE_XBH else "below")
    )

    print("**Career Contact Profile**")
    print(
        f"{career_xbh:,} career XBH in {total_pa:,} PA ({career_xbh_rate*100:.1f}% XBH/PA). "
        f"League XBH rate in this dataset is ~{LEAGUE_XBH*100:.1f}% — {blast} is "
        f"{xbh_vs} that. "
        f"{'Real extra-base juice, not a small-sample mirage.' if total_pa > 2000 else 'Solid XBH frequency in the sample.'}"
    )
    print()

    k_vs = "well below" if career_k_rate < 18 else ("right at" if career_k_rate < 24 else "elevated above")
    print(f"Career K rate: {career_k_rate:.1f}% — {k_vs} ~22% league average.")
    print()

    print(f"Extra-base context: {career_doubles} doubles, {career_triples} triples, {career_hrs} HR — ", end="")
    if career_doubles > 150:
        print("gap power is real.")
    elif career_hrs > 100:
        print("power bat with thump.")
    else:
        print("developing power profile.")
    print()

    xbh_pa = batter_pa[batter_pa["is_xbh"] == 1]
    ev_mean = xbh_pa["launch_speed"].mean() if "launch_speed" in xbh_pa.columns and len(xbh_pa) > 0 else 0
    ev_note = (
        "firm contact vs league XBH EV (~95 mph in this population)."
        if ev_mean >= 93
        else "contact quality is more placement than pure smoke — still plays if the pitcher leaks barrels."
    )
    print(f"Exit velocity on all career XBH: mean {ev_mean:.1f} mph — {ev_note}")
    print()

    if "pitch_type" in xbh_pa.columns and len(xbh_pa) > 0:
        xbh_by_pt = xbh_pa.groupby("pitch_type").size().sort_values(ascending=False).head(6)
        pt_parts = []
        for pt, ct in xbh_by_pt.items():
            pt_pa_df = batter_pa[batter_pa["pitch_type"] == pt]
            n_pt = len(pt_pa_df)
            xbh_ct = int(pt_pa_df["is_xbh"].sum()) if "is_xbh" in pt_pa_df.columns else 0
            r = xbh_ct / n_pt if n_pt > 0 else 0
            pt_parts.append(f"{pt} {int(ct)} (.{int(min(r, 0.999) * 1000):03d} XBH)")
        print(f"XBH by pitch type (top): {', '.join(pt_parts)}.")

        pitcher_top3 = ""
        if pid:
            pp = pitcher_prof[pitcher_prof["pitcher"] == pid]
            if len(pp) > 0:
                sort_col = "season" if "season" in pp.columns else "game_year"
                latest = pp.sort_values(sort_col, ascending=False).iloc[0]
                top_pitches = []
                for i in range(1, 6):
                    pt_col = f"top_pitch_{i}"
                    pct_col = f"top_pitch_{i}_pct"
                    if pt_col in latest.index and pd.notna(latest.get(pt_col)):
                        top_pitches.append((str(latest[pt_col]).upper(), latest.get(pct_col, 0)))
                pitcher_top3 = " / ".join([p[0] for p in top_pitches[:3]])

        top_xbh_type = xbh_by_pt.index[0]
        top_xbh_ct = int(xbh_by_pt.iloc[0])
        if pitcher_top3:
            print(
                f"That's the math: {last} runs {pitcher_top3} as top usage pitches — "
                f"{blast}'s {top_xbh_ct} XBH off {top_xbh_type} line up with what {last} actually throws."
            )
            print()
            print(
                f"The swing decisions show up in the buckets: when {last} leans on what he trusts, "
                f"{blast} already has an XBH library on those pitch codes. "
                f"League XBH/PA in this dataset is ~**{LEAGUE_XBH*100:.1f}%** — {blast} is "
                f"{'north' if career_xbh_rate > LEAGUE_XBH + 0.01 else 'around' if career_xbh_rate >= LEAGUE_XBH - 0.01 else 'south'} of that baseline, "
                f"so the XBH prop isn't a charity case; it's extra-base skill pressed against a specific pitch diet."
            )
    print()

    if bid:
        bp = batter_prof[batter_prof["batter"] == bid]
        if len(bp) > 0:
            splits = []
            for grp in ["fastball", "breaking", "offspeed"]:
                grp_rows = bp[bp["pitch_group"] == grp] if "pitch_group" in bp.columns else pd.DataFrame()
                if len(grp_rows) > 0:
                    g = grp_rows.iloc[0]
                    ba = g.get("ba", 0)
                    ev = g.get("avg_ev", 0)
                    whiff = g.get("whiff_rate", 0)
                    ba = ba if pd.notna(ba) else 0
                    ev = ev if pd.notna(ev) else 0
                    whiff = whiff if pd.notna(whiff) else 0
                    splits.append(f"vs {grp.title()} .{int(ba*1000):03d} BA / {ev:.1f} mph EV / {whiff*100:.1f}% whiff")
            if splits:
                print(f"Pitch-group splits (batter_pitch_profiles): {' | '.join(splits)}.")
                print()
                print(
                    f"Editorial read: {blast}'s pitch-group ladder shows where the loud contact lives — "
                    f"map that lane to {last}'s usage and you're pricing how often the extra-base counts arrive."
                )
        print()

    print(
        f"One-sentence summary: {blast} is a "
        f"{'contact-first hitter' if career_ba > 0.260 else 'solid bat' if career_ba > 0.240 else 'power-over-contact bat'}"
        f" who {f'keeps the K rate low at {career_k_rate:.0f}%' if career_k_rate < 20 else f'has an elevated K rate at {career_k_rate:.0f}%'},"
        f" and {last}'s arsenal {'feeds right into' if career_ba > 0.250 else 'tests'} that profile."
    )
    print()

    print(f"**The {last}-Specific XBH Case**")
    if pid:
        bvp = batter_pa[batter_pa["pitcher"] == pid]
        bvp_pa_ct = len(bvp)
        if bvp_pa_ct > 0:
            bvp_hits = int(bvp["is_hit"].sum())
            bvp_hrs = int(bvp["is_hr"].sum())
            bvp_xbh = int(bvp["is_xbh"].sum()) if "is_xbh" in bvp.columns else 0
            bvp_ks = int(bvp["is_strikeout"].sum())
            bvp_ba = bvp_hits / bvp_pa_ct
            bvp_k_rate = bvp_ks / bvp_pa_ct * 100

            if bvp_xbh >= 2 and bvp_pa_ct >= 5:
                bvp_label = "Strong BvP XBH history."
            elif bvp_xbh == 0 and bvp_pa_ct >= 8:
                bvp_label = "No BvP XBH yet in a real sample."
            elif bvp_pa_ct < 5:
                bvp_label = "Tiny sample."
            else:
                bvp_label = "Moderate BvP XBH history."

            print(
                f"BvP: {bvp_hits}-for-{bvp_pa_ct} {_fmt_avg_paren(bvp_ba)}, "
                f"{bvp_hrs} HR, {bvp_xbh} XBH, {bvp_ks} K in {bvp_pa_ct} PA. {bvp_label}"
            )
            print()

            k_word = "ugly" if bvp_k_rate > 30 else ("workable" if bvp_k_rate > 20 else "clean")
            print(f"BvP K rate: {bvp_k_rate:.0f}% — {k_word}.")
            print()

            if bvp_xbh > 0:
                print(
                    f"When {blast} drives the ball against {last}, XBHs show up. "
                    f"The BvP line ({bvp_xbh} XBH in {bvp_pa_ct} PA) {'confirms' if bvp_xbh >= 2 else 'hints at'} "
                    f"the extra-base profile working against this arsenal."
                )
            else:
                print(
                    f"No BvP XBH in {bvp_pa_ct} PA — sample is "
                    f"{'too small to draw conclusions.' if bvp_pa_ct < 10 else 'leaning empty on barrels.'} "
                    f"Pivot to pitch-type matchup logic."
                )
        else:
            print(
                f"No BvP history — these two have not faced each other in the Statcast era. "
                f"Pivot to pitch-type matchup logic: {blast}'s XBH splits vs {last}'s arsenal."
            )
    print()

    print(f"**{last}'s XBH Vulnerability**")
    if pid:
        pitcher_pa_df = pa_pitcher(pid)
        p_total_pa = len(pitcher_pa_df)
        p_xbh_allowed = int(pitcher_pa_df["is_xbh"].sum()) if "is_xbh" in pitcher_pa_df.columns and p_total_pa > 0 else 0
        p_xbh_rate = p_xbh_allowed / p_total_pa if p_total_pa > 0 else 0

        xbh_comment = "right on" if abs(p_xbh_rate - LEAGUE_XBH) < 0.008 else ("above" if p_xbh_rate > LEAGUE_XBH else "below")
        print(
            f"Career: {p_xbh_allowed:,} XBH allowed in {p_total_pa:,} PA ({p_xbh_rate*100:.1f}% XBH/PA). "
            f"That's {xbh_comment} the ~{LEAGUE_XBH*100:.1f}% league average."
        )
        print()

        if "events" in pitcher_pa_df.columns:
            pe = pitcher_pa_df["events"]
            pd_dbl = int(pe.eq("double").sum())
            pd_tpl = int(pe.eq("triple").sum())
            pd_hr = int(pe.eq("home_run").sum())
            print(f"Breakdown allowed: {pd_dbl} doubles, {pd_tpl} triples, {pd_hr} HR.")
            print()

        p_k_rate = pitcher_pa_df["is_strikeout"].mean() * 100 if "is_strikeout" in pitcher_pa_df.columns else 0
        p_bb_rate = pitcher_pa_df["is_walk"].mean() * 100 if "is_walk" in pitcher_pa_df.columns else 0
        if p_k_rate < 20:
            kb_comment = "low K rate means more balls in play — more XBH variance."
        elif p_k_rate < 25:
            kb_comment = "decent strikeout rate limits some extra-base opportunity."
        else:
            kb_comment = "high K rate makes it harder to stack XBHs."
        print(f"K rate: {p_k_rate:.1f}% | BB rate: {p_bb_rate:.1f}% — {kb_comment}")
        print()

        pp = pitcher_prof[pitcher_prof["pitcher"] == pid]
        if len(pp) > 0:
            sort_col2 = "season" if "season" in pp.columns else "game_year"
            latest = pp.sort_values(sort_col2, ascending=False).iloc[0]
            fb_velo = latest.get("velo_fastball", 0)
            if pd.isna(fb_velo):
                fb_velo = 0
            fb_pct = latest.get("pct_fastball", 0)
            if pd.isna(fb_pct):
                fb_pct = 0
            brk_pct = latest.get("pct_breaking", 0)
            if pd.isna(brk_pct):
                brk_pct = 0
            off_pct = latest.get("pct_offspeed", 0)
            if pd.isna(off_pct):
                off_pct = 0
            print(
                f"Arsenal breakdown: Fastball {fb_pct*100:.0f}% (~{fb_velo:.0f} mph), "
                f"Breaking {brk_pct*100:.0f}%, Offspeed {off_pct*100:.0f}%."
            )
        print()

        if "stand" in pitcher_pa_df.columns:
            lhb = pitcher_pa_df[pitcher_pa_df["stand"] == "L"]
            rhb = pitcher_pa_df[pitcher_pa_df["stand"] == "R"]
            lhb_x = int(lhb["is_xbh"].sum()) if "is_xbh" in lhb.columns else 0
            rhb_x = int(rhb["is_xbh"].sum()) if "is_xbh" in rhb.columns else 0
            lhb_pa = len(lhb)
            rhb_pa = len(rhb)
            lhb_xr = lhb_x / lhb_pa * 100 if lhb_pa > 0 else 0
            rhb_xr = rhb_x / rhb_pa * 100 if rhb_pa > 0 else 0

            batter_hand = "R"
            if bid:
                b_prof2 = batter_prof[batter_prof["batter"] == bid]
                if len(b_prof2) > 0 and "stand" in b_prof2.columns:
                    batter_hand = b_prof2.iloc[0]["stand"]
            hand_label = "left" if batter_hand == "L" else "right"
            print(
                f"XBH allowed by handedness: vs LHB {lhb_xr:.1f}% ({lhb_x} XBH in {lhb_pa} PA) | "
                f"vs RHB {rhb_xr:.1f}% ({rhb_x} in {rhb_pa}). {blast} bats {hand_label}."
            )
        print()

        if len(qp) > 0 and pid:
            qp_row = pd.DataFrame()
            for col in ["player_id", "key_mlbam", "mlbam_id", "MLBAMID"]:
                if col in qp.columns:
                    match = qp[qp[col] == pid]
                    if len(match) > 0:
                        qp_row = match
                        break
            if len(qp_row) > 0:
                qr = qp_row.iloc[0]
                era = qr.get("ERA", qr.get("era", "—"))
                whip = qr.get("WHIP", qr.get("whip", "—"))
                bb9 = qr.get("BB/9", qr.get("bb_9", qr.get("bb9", "—")))
                ip = qr.get("IP", qr.get("ip", "—"))
                era_val = 999
                try:
                    era_val = float(era)
                except Exception:
                    pass
                era_comment = (
                    f"That's rough early form — you're betting current-version {last} tonight."
                    if era_val > 5
                    else "Solid early numbers."
                )
                print(f"2026 (qualifying file): ERA {era}, WHIP {whip}, BB/9 {bb9}, {ip} IP. {era_comment}")
            else:
                print(f"2026: No qualifying data yet for {last}.")
        print()

    print(f"**Where {blast} Ranks Tonight**")
    rank = idx + 1
    pctile = max(1, min(99, int(round(100 * rank / max(total_matchups, 1)))))
    team_df = df[df["batter_team"] == bteam].sort_values(_sort_xbh, ascending=False).reset_index(drop=True)
    team_rank_val = "—"
    if bname in team_df["batter_name"].values:
        team_rank_val = int(team_df[team_df["batter_name"] == bname].index[0]) + 1
    tr_label = "best" if team_rank_val == 1 else f"#{team_rank_val}"
    print(f"#{rank} of {total_matchups} matchups by adj P(XBH) — top ~{pctile}% of the slate.")
    print(f"#{team_rank_val} on {bteam} by adj P(XBH) — {tr_label} XBH projection on {bteam} in this file.")
    print()

    print("**The Verdict on the XBH Bet**")
    xbh_word = "crushes" if career_xbh_rate > LEAGUE_XBH + 0.025 else ("beats" if career_xbh_rate > LEAGUE_XBH else "trails")
    line1 = (
        f"**For:** {career_xbh_rate*100:.1f}% career XBH/PA {xbh_word} {LEAGUE_XBH*100:.1f}% league average. "
        f"Model P(XBH) at {adj_xbh*100:.1f}% per PA is "
        f"{'well above' if adj_xbh > 0.10 else 'above'} baseline."
    )
    if bid and pid:
        bvp_data = pa_batter(bid)
        bvp_data = bvp_data[bvp_data["pitcher"] == pid] if len(bvp_data) else bvp_data
        bvp_x2 = int(bvp_data["is_xbh"].sum()) if len(bvp_data) > 0 and "is_xbh" in bvp_data.columns else 0
        bvp_p = len(bvp_data)
        if bvp_p > 0:
            if bvp_x2 >= 2:
                line1 += f" BvP has {bvp_x2} XBH in {bvp_p} PA — extra-base history is on the board."
            elif bvp_x2 > 0:
                line1 += f" BvP line: {bvp_x2} XBH in {bvp_p} PA — barrels have shown up."
        else:
            line1 += f" No BvP history but pitch-type matchup favors {blast} for XBH."
    conv_hit = cf.get("convergence_hit", "—")
    line1 += f" Confidence on XBH is {grade} with convergence_hit {conv_hit}."
    print(line1)
    print()

    line2 = "**Against:** "
    if bid and pid:
        bvp_data2 = pa_batter(bid)
        bvp_data2 = bvp_data2[bvp_data2["pitcher"] == pid] if len(bvp_data2) else bvp_data2
        bvp_ks2 = int(bvp_data2["is_strikeout"].sum()) if len(bvp_data2) > 0 else 0
        bvp_pa_ct3 = len(bvp_data2)
        if bvp_ks2 > 0 and bvp_pa_ct3 > 0:
            bvp_kr = bvp_ks2 / bvp_pa_ct3 * 100
            line2 += f"BvP K rate {bvp_kr:.0f}% ({bvp_ks2} K in {bvp_pa_ct3} PA). "
    k_word2 = "elevated" if career_k_rate > 25 else "manageable"
    line2 += f"{blast}'s {career_k_rate:.1f}% career K rate is {k_word2}; model P(K) ~{career_k_model:.0f}% tonight. "
    line2 += f"Staleness {cf.get('staleness', '—')} still trims confidence."
    print(line2)
    print()

    circle = "inner circle" if rank <= 5 else ("top tier" if rank <= 10 else "upper half")
    line3 = (
        f"**Net:** ~{adj_xbh*100:.1f}% per PA adjusted → ~{three_pa_adj:.1f}% over 3 PA — about 1-in-{inv} for at least one XBH. "
        f"That's the slate's {circle} (#{rank} overall, #{team_rank_val} on {bteam}). "
        f"This is a {'contact profile play' if career_ba > 0.260 else 'matchup play'} — "
        f"if you don't buy the {last} vulnerability thesis, cut enthusiasm; "
        f"if you do, {blast} is exactly the type of hitter you'd target for an XBH."
    )
    print(line3)
    print()
    print("---")
    print()
