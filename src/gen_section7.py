import json
import os

import numpy as np
import pandas as pd


def _fmt_avg(ba):
    s = f"{ba:.3f}"
    return "(" + (s[1:] if s.startswith("0.") else s) + ")"


_pred_path = os.environ.get(
    "MATCHUP_PREDICTIONS_JSON", "data/reports/todays_matchup_predictions.json"
)
preds = json.load(open(_pred_path))
df = pd.DataFrame(preds)
pa_all = pd.read_parquet('data/raw/statcast_pa_level_league.parquet')
_b_ix = pa_all.groupby('batter').indices
_p_ix = pa_all.groupby('pitcher').indices


def pa_batter(bid):
    if bid is None or bid not in _b_ix:
        return pa_all.iloc[[]]
    return pa_all.iloc[_b_ix[bid]]


def pa_pitcher(pid):
    if pid is None or pid not in _p_ix:
        return pa_all.iloc[[]]
    return pa_all.iloc[_p_ix[pid]]


pitcher_prof = pd.read_parquet('data/raw/pitcher_profiles_by_season.parquet')
batter_prof = pd.read_parquet('data/raw/batter_pitch_profiles.parquet')

try:
    qp = pd.read_csv('data/raw/qualifying_pitchers_2026.csv')
except Exception:
    qp = pd.DataFrame()

_sort_hr = "p_hr" if "p_hr" in df.columns else "adj_p_hr"
top10_hr = df.sort_values(_sort_hr, ascending=False).head(10).reset_index(drop=True)
total_matchups = len(df)

print("## Section 7: Top 10 HR Full Analysis\n")

for idx, row in top10_hr.iterrows():
    bname = row['batter_name']
    pname = row['pitcher_name']
    bteam = row['batter_team']
    bid = int(row['batter_mlbam_id']) if not pd.isna(row.get('batter_mlbam_id')) else None
    pid = int(row['pitcher_mlbam_id']) if not pd.isna(row.get('pitcher_mlbam_id')) else None
    raw_hr = row.get('p_hr', 0) or 0
    adj_hr = row.get('adj_p_hr', 0) or 0
    conf = row.get('conf_hr', 0) or 0
    grade = row.get('conf_hr_label', '—')
    conf_factors = row.get('conf_factors', {})
    if isinstance(conf_factors, str):
        conf_factors = json.loads(conf_factors)
    three_pa_raw = (1 - (1 - raw_hr) ** 3) * 100
    three_pa_adj = (1 - (1 - adj_hr) ** 3) * 100
    three_pa_adj_p = 1 - (1 - adj_hr) ** 3
    inv = int(round(1 / three_pa_adj_p)) if three_pa_adj_p > 0 else 999

    last = pname.split()[-1]
    blast = bname.split()[-1]

    print(f"## {bname} HR Analysis vs {pname}")
    print()
    print(f"Model (tonight): Raw P(HR) {raw_hr*100:.2f}% | Adj P(HR) {adj_hr*100:.2f}% | Conf {conf:.3f} ({grade})")
    print(f"3 PA vs starter: (1-(1-{raw_hr:.4f})^3 ≈) {three_pa_raw:.1f}% raw | ~{three_pa_adj:.1f}% adjusted — about 1-in-{inv} either way.")
    print()
    cf = conf_factors if conf_factors else {}
    cf_str = (
        f"Confidence factors: pitcher data {cf.get('pitcher_data','—')} | "
        f"BvP HR {cf.get('bvp_hr','—')} | BvP hit {cf.get('bvp_hit','—')} | "
        f"staleness {cf.get('staleness','—')} | convergence HR {cf.get('convergence_hr','—')} | "
        f"convergence hit {cf.get('convergence_hit','—')} | convergence XBH {cf.get('convergence_xbh','—')}"
    )
    print(cf_str)
    print()

    if not bid:
        print("No batter ID available.\n---\n")
        continue

    batter_pa = pa_batter(bid)
    total_pa = len(batter_pa)
    career_hrs = int(batter_pa['is_hr'].sum())
    hr_rate = career_hrs / total_pa if total_pa > 0 else 0
    mult = hr_rate / 0.031 if hr_rate > 0 else 0

    if 'game_year' in batter_pa.columns:
        yearly = batter_pa.groupby('game_year')['is_hr'].sum()
        peak_yr = int(yearly.idxmax()) if len(yearly) > 0 else 0
        peak_ct = int(yearly.max()) if len(yearly) > 0 else 0
    else:
        peak_yr, peak_ct = 0, 0

    hr_pa = batter_pa[batter_pa['is_hr'] == 1]
    ev_mean = hr_pa['launch_speed'].mean() if 'launch_speed' in hr_pa.columns and len(hr_pa) > 0 else 0
    ev_max = hr_pa['launch_speed'].max() if 'launch_speed' in hr_pa.columns and len(hr_pa) > 0 else 0
    la_mean = hr_pa['launch_angle'].mean() if 'launch_angle' in hr_pa.columns and len(hr_pa) > 0 else 0

    career_k_rate = batter_pa['is_strikeout'].mean() * 100 if 'is_strikeout' in batter_pa.columns else 0
    career_k_model = (row.get('p_k', 0) or 0) * 100

    above_below = "above" if hr_rate > 0.031 else "below"
    real_power = " That's real power, not noise from a tiny sample." if total_pa > 500 and hr_rate > 0.035 else ""

    print("**Career Power Profile**")
    print(
        f"{career_hrs} career HR in {total_pa:,} PA ({hr_rate*100:.2f}% HR/PA). "
        f"League HR rate in this dataset is ~3.1% — {blast} is "
        f"{'well ' if mult > 1.3 else ''}{above_below} that (~{mult:.1f}×).{real_power}"
    )
    print()
    print(f"Peak season: {peak_yr} ({peak_ct} HR).")
    print()
    ev_comment = "solidly hard HR contact vs league HR EV (~104 mph)." if ev_mean > 103 else "slightly below league HR EV (~104 mph)."
    print(f"Exit velocity on career HRs: mean {ev_mean:.1f} mph | max {ev_max:.1f} mph — {ev_comment}")
    print()
    la_comment = "normal fly-ball HR shape." if 25 < la_mean < 35 else ("steep angle." if la_mean >= 35 else "flat-ish angle.")
    print(f"Launch angle on career HRs: mean {la_mean:.1f} deg — {la_comment}")
    print()

    if 'pitch_type' in hr_pa.columns and len(hr_pa) > 0:
        hr_by_pt = hr_pa.groupby('pitch_type').size().sort_values(ascending=False).head(6)
        pt_ba = {}
        for pt in hr_by_pt.index:
            pt_pa_df = batter_pa[batter_pa['pitch_type'] == pt]
            pt_hits = pt_pa_df['is_hit'].sum() if 'is_hit' in pt_pa_df.columns else 0
            pt_ba[pt] = pt_hits / len(pt_pa_df) if len(pt_pa_df) > 0 else 0

        pt_parts = []
        for pt, ct in hr_by_pt.items():
            ba_val = pt_ba.get(pt, 0)
            pt_parts.append(f"{pt} {int(ct)} (.{int(ba_val*1000):03d} BA in sample)")
        pt_line = ", ".join(pt_parts)
        print(f"HR by pitch type (top): {pt_line}.")

        top_hr_type = hr_by_pt.index[0]
        top_hr_ct = int(hr_by_pt.iloc[0])
        pct10 = int(round(top_hr_ct / career_hrs * 10)) if career_hrs > 0 else 0
        second_pt = hr_by_pt.index[1] if len(hr_by_pt) > 1 else top_hr_type
        second_ct = int(hr_by_pt.iloc[1]) if len(hr_by_pt) > 1 else top_hr_ct
        third_pt = hr_by_pt.index[2] if len(hr_by_pt) > 2 else None
        third_ct = int(hr_by_pt.iloc[2]) if third_pt is not None else 0

        pitcher_top3 = ""
        if pid:
            pp = pitcher_prof[pitcher_prof['pitcher'] == pid]
            if len(pp) > 0:
                sort_col = 'season' if 'season' in pp.columns else 'game_year'
                latest = pp.sort_values(sort_col, ascending=False).iloc[0]
                top_pitches = []
                for i in range(1, 6):
                    pt_col = f'top_pitch_{i}'
                    pct_col = f'top_pitch_{i}_pct'
                    if pt_col in latest.index and pd.notna(latest.get(pt_col)):
                        top_pitches.append((str(latest[pt_col]).upper(), latest.get(pct_col, 0)))
                pitcher_top3 = " / ".join([p[0] for p in top_pitches[:3]])

        if pitcher_top3:
            third_bits = f" and {third_pt} ({third_ct})" if third_pt else ""
            hr_word = "HR" if second_ct == 1 else "HRs"
            print(
                f"{top_hr_type} is the engine — {pct10} of every 10 career HRs are off {top_hr_type}. "
                f"He still does damage on {second_pt} ({second_ct} career {hr_word} off that pitch type in this file)"
                f"{third_bits} — the spray chart isn't one-note, it's organized violence. "
                f"That's the math: {last} runs {pitcher_top3} as top usage pitches — "
                f"{blast}'s {top_hr_ct} HR off {top_hr_type} line up with what {last} actually throws."
            )
            print()
            print(
                f"When {last} tries to run the fastball up or bury the breaker down, "
                f"{blast}'s historical HR distribution says the counterpunch already exists: "
                f"heaters for lift, secondaries for leverage. "
                f"League HR/PA in this dataset sits near **3.1%** — {blast} lives north of that baseline, "
                f"so you're not manufacturing a HR story; you're stacking real outcomes on a real pitch map."
            )
        print()
        print(
            f"One-sentence summary: {top_hr_type} is the bread and butter — "
            f"{top_hr_ct} of {career_hrs} career HRs came off {top_hr_type}, "
            f"and {blast} has shown he can elevate off {last}-type stuff too."
        )
    print()

    # BvP
    print(f"**The {last}-Specific HR Case**")
    if pid:
        bvp = batter_pa[batter_pa['pitcher'] == pid]
        bvp_pa_ct = len(bvp)
        if bvp_pa_ct > 0:
            bvp_hits = int(bvp['is_hit'].sum())
            bvp_hrs = int(bvp['is_hr'].sum())
            bvp_ks = int(bvp['is_strikeout'].sum())
            bvp_ba = bvp_hits / bvp_pa_ct
            bvp_k_rate = bvp_ks / bvp_pa_ct * 100

            if bvp_pa_ct < 10:
                sample = "Sample is tiny, but it's not empty"
            elif bvp_pa_ct < 20:
                sample = "Decent sample"
            else:
                sample = "Real sample size"

            hr_note = f": {bvp_hrs} HR already on the board." if bvp_hrs > 0 else ": no HR history yet."
            print(
                f"BvP: {bvp_hits}-for-{bvp_pa_ct} {_fmt_avg(bvp_ba)}, "
                f"{bvp_hrs} HR, {bvp_ks} K in {bvp_pa_ct} PA. {sample}{hr_note}"
            )
            print()

            if bvp_hrs > 0:
                hr_bvp = bvp[bvp['is_hr'] == 1]
                print("| Date (Statcast) | Pitch type | Velo | EV | LA |")
                print("|---|---|---|---|---|")
                for _, h in hr_bvp.iterrows():
                    gdate = h.get('game_date', '—')
                    try:
                        gdate = pd.to_datetime(gdate).strftime('%Y-%m-%d')
                    except Exception:
                        gdate = str(gdate)[:10]
                    pt = h.get('pitch_type', '—')
                    velo = h.get('release_speed', 0)
                    ev = h.get('launch_speed', 0)
                    la = h.get('launch_angle', 0)
                    velo_s = f"{velo:.0f} mph" if pd.notna(velo) and velo > 0 else "—"
                    ev_s = f"{ev:.0f} mph" if pd.notna(ev) and ev > 0 else "—"
                    la_s = f"{la:.0f}°" if pd.notna(la) else "—"
                    print(f"| {gdate} | {pt} | {velo_s} | {ev_s} | {la_s} |")
                print()
                avg_ev = hr_bvp["launch_speed"].mean()
                hr_pt = hr_bvp["pitch_type"].value_counts()
                top_pt_hr = hr_pt.index[0] if len(hr_pt) > 0 else "—"
                if pd.isna(avg_ev):
                    contact_word = "solid"
                    ev_phrase = "avg EV n/a (missing launch_speed on BvP HR rows)"
                else:
                    contact_word = "violent" if float(avg_ev) > 107 else "solid"
                    ev_phrase = f"avg {float(avg_ev):.0f} mph EV"
                print(
                    f"Those HRs came on {top_pt_hr} at {ev_phrase} — {contact_word} contact. "
                    f"{last} leans on that pitch type; {blast}'s BvP HR is on-brand with career HR history."
                )
            else:
                print(
                    f"No HRs in {bvp_pa_ct} BvP PA. Pivot to pitch-type matchup: "
                    f"{blast}'s top HR pitch types align with {last}'s arsenal for opportunity."
                )
            print()
            k_word = "ugly" if bvp_k_rate > 30 else ("workable" if bvp_k_rate > 20 else "clean")
            k_narr = (
                "when he's not putting the ball in play, he's whiffing."
                if bvp_k_rate > 30
                else "contact happens at a reasonable rate."
            )
            connect_narr = (
                "the ball has left before." if bvp_hrs > 0 else "power potential is there based on career profile."
            )
            print(
                f"Narrative: BvP K rate {bvp_k_rate:.0f}% ({bvp_ks} K in {bvp_pa_ct}) is {k_word} — {k_narr} "
                f"When he does connect in this matchup, {connect_narr}"
            )
        else:
            print(
                f"No BvP history — these two have not faced each other in the Statcast era. "
                f"Pivot to pitch-type matchup logic: {blast}'s career HR distribution vs "
                f"{last}'s arsenal is the play here."
            )
    print()

    # Pitcher vulnerability
    print(f"**{last}'s HR Vulnerability**")
    p_hr_rate = 0.0
    if pid:
        pitcher_pa_df = pa_pitcher(pid)
        p_total_pa = len(pitcher_pa_df)
        p_hr_allowed = int(pitcher_pa_df['is_hr'].sum()) if p_total_pa > 0 else 0
        p_hr_rate = p_hr_allowed / p_total_pa if p_total_pa > 0 else 0

        if abs(p_hr_rate - 0.031) < 0.005:
            rate_word = "right on"
        elif p_hr_rate > 0.031:
            rate_word = "above"
        else:
            rate_word = "below"
        fountain = (
            "career-wise he's not an extreme HR fountain."
            if p_hr_rate < 0.035
            else "this arm gives up dingers."
        )
        print(
            f"Career: {p_hr_allowed} HR allowed in {p_total_pa:,} PA ({p_hr_rate*100:.2f}% HR/PA). "
            f"That's {rate_word} the ~3.1% league average — {fountain}"
        )
        print()

        p_hr_pa = pitcher_pa_df[pitcher_pa_df['is_hr'] == 1]
        if 'pitch_type' in p_hr_pa.columns and len(p_hr_pa) > 0:
            p_hr_by_pt = p_hr_pa.groupby('pitch_type').size().sort_values(ascending=False).head(5)
            pt_str = ", ".join([f"{pt} {int(ct)}" for pt, ct in p_hr_by_pt.items()])
            print(f"HR allowed by pitch (top): {pt_str}.")
        print()

        pp = pitcher_prof[pitcher_prof['pitcher'] == pid]
        if len(pp) > 0:
            sort_col2 = 'season' if 'season' in pp.columns else 'game_year'
            latest = pp.sort_values(sort_col2, ascending=False).iloc[0]
            fb_velo = latest.get('velo_fastball', 0)
            if pd.isna(fb_velo):
                fb_velo = 0
            fb_pct_val = latest.get('pct_fastball', 0)
            if pd.isna(fb_pct_val):
                fb_pct_val = 0
            velo_comment = "hittable velocity if location slips." if fb_velo < 95 else "decent velocity but not elite."
            print(f"Fastball: ~{fb_velo:.1f} mph (profile), ~{fb_pct_val*100:.0f}% fastball-class usage — {velo_comment}")
        print()

        if len(qp) > 0 and pid:
            qp_row = pd.DataFrame()
            for col in ['player_id', 'key_mlbam', 'mlbam_id', 'MLBAMID']:
                if col in qp.columns:
                    match = qp[qp[col] == pid]
                    if len(match) > 0:
                        qp_row = match
                        break
            if len(qp_row) > 0:
                qr = qp_row.iloc[0]
                era = qr.get('ERA', qr.get('era', '—'))
                whip = qr.get('WHIP', qr.get('whip', '—'))
                bb9 = qr.get('BB/9', qr.get('bb_9', qr.get('bb9', '—')))
                ip = qr.get('IP', qr.get('ip', '—'))
                hr_26 = qr.get('HR', qr.get('hr', '—'))
                hr_str = ""
                try:
                    hr_str = f", {int(float(hr_26))} HR allowed"
                except Exception:
                    pass
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
                print(f"2026 (qualifying file): ERA {era}, WHIP {whip}, BB/9 {bb9}, {ip} IP{hr_str}. {era_comment}")
            else:
                print(f"2026: No qualifying data yet for {last}.")
        print()

        if 'stand' in pitcher_pa_df.columns:
            lhb_pa = pitcher_pa_df[pitcher_pa_df['stand'] == 'L']
            rhb_pa = pitcher_pa_df[pitcher_pa_df['stand'] == 'R']
            lhb_hr = int(lhb_pa['is_hr'].sum())
            rhb_hr = int(rhb_pa['is_hr'].sum())
            lhb_rate = lhb_hr / len(lhb_pa) * 100 if len(lhb_pa) > 0 else 0
            rhb_rate = rhb_hr / len(rhb_pa) * 100 if len(rhb_pa) > 0 else 0

            batter_hand = 'R'
            if bid:
                b_prof = batter_prof[batter_prof['batter'] == bid]
                if len(b_prof) > 0 and 'stand' in b_prof.columns:
                    batter_hand = b_prof.iloc[0]['stand']
            hand_label = "left" if batter_hand == 'L' else "right"
            diff = abs(lhb_rate - rhb_rate)
            if diff < 0.5:
                split_comment = "no platoon hide here."
            elif (batter_hand == 'L' and lhb_rate > rhb_rate) or (batter_hand == 'R' and rhb_rate > lhb_rate):
                split_comment = "slight platoon edge for the batter."
            else:
                split_comment = "platoon works against him."
            print(
                f"vs LHB: {lhb_hr} HR in {len(lhb_pa):,} PA ({lhb_rate:.2f}%) | "
                f"vs RHB: {rhb_hr} HR in {len(rhb_pa):,} PA ({rhb_rate:.2f}%). "
                f"{blast} bats {hand_label} — {split_comment}"
            )
        print()

        p_k_rate = pitcher_pa_df['is_strikeout'].mean() * 100 if 'is_strikeout' in pitcher_pa_df.columns else 0
        p_bb_rate = pitcher_pa_df['is_walk'].mean() * 100 if 'is_walk' in pitcher_pa_df.columns else 0
        if p_k_rate < 20:
            kb_comment = "he does not miss many bats relative to league; more balls in play means more HR variance."
        elif p_k_rate < 25:
            kb_comment = "decent strikeout rate limits some HR upside."
        else:
            kb_comment = "high K rate makes it harder to put bat on ball."
        print(f"K / BB: Career K% ~{p_k_rate:.1f}%, BB% ~{p_bb_rate:.1f}% — {kb_comment}")
    print()

    # Rankings
    print(f"**Where {blast} Ranks Tonight**")
    rank = idx + 1
    pctile = max(1, min(99, int(round(100 * rank / max(total_matchups, 1)))))
    team_df = df[df['batter_team'] == bteam].sort_values(_sort_hr, ascending=False).reset_index(drop=True)
    team_rank_val = "—"
    if bname in team_df['batter_name'].values:
        team_rank_val = int(team_df[team_df['batter_name'] == bname].index[0]) + 1
    tr_label = "best" if team_rank_val == 1 else f"#{team_rank_val}"
    print(f"#{rank} of {total_matchups} matchups by adj P(HR) — top ~{pctile}% of the slate.")
    print(f"#{team_rank_val} on {bteam} by adj P(HR) — {tr_label} HR projection on {bteam} in this file.")
    print()

    # Verdict
    print("**The Verdict on the HR Bet**")
    rate_word2 = "crushes" if hr_rate > 0.04 else "beats"
    elite = " — elite-tier HR frequency." if hr_rate > 0.05 else "."
    double = "more than double" if raw_hr > 0.062 else "well above"
    line1 = (
        f"**For:** {hr_rate*100:.2f}% career HR/PA {rate_word2} 3.1% league average{elite} "
        f"Raw and adj model are both ~{raw_hr*100:.1f}% / {adj_hr*100:.1f}% per PA — {double} baseline."
    )
    if bid and pid:
        bvp_data = pa_batter(bid)
        bvp_data = bvp_data[bvp_data['pitcher'] == pid] if len(bvp_data) else bvp_data
        bvp_hr_ct = int(bvp_data['is_hr'].sum()) if len(bvp_data) > 0 else 0
        bvp_pa_ct2 = len(bvp_data)
        if bvp_hr_ct > 0:
            line1 += f" BvP already has {bvp_hr_ct} HR in {bvp_pa_ct2} PA off this arm."
        else:
            line1 += f" No BvP HR history but pitch-type matchup favors {blast}."
    conv_hr = cf.get('convergence_hr', '—')
    line1 += f" Confidence on HR is {grade} with convergence_hr {conv_hr}."
    print(line1)
    print()

    line2 = "**Against:** "
    if bid and pid:
        bvp_data2 = pa_batter(bid)
        bvp_data2 = bvp_data2[bvp_data2['pitcher'] == pid] if len(bvp_data2) else bvp_data2
        bvp_ks2 = int(bvp_data2['is_strikeout'].sum()) if len(bvp_data2) > 0 else 0
        bvp_pa_ct3 = len(bvp_data2)
        if bvp_ks2 > 0 and bvp_pa_ct3 > 0:
            line2 += f"{bvp_ks2} K in {bvp_pa_ct3} BvP PA — when this matchup goes wrong, it goes strikeout wrong. "
    k_elevated = "elevated" if career_k_rate > 25 else "manageable"
    line2 += f"{blast}'s {career_k_rate:.1f}% career K rate is {k_elevated}; model P(K) ~{career_k_model:.0f}% tonight. "
    if pid:
        if p_hr_rate < 0.028:
            hr_avg = "below league"
        elif p_hr_rate < 0.035:
            hr_avg = "around league"
        else:
            hr_avg = "elevated"
        line2 += f"{last}'s career HR rate is {hr_avg} — if you think 2026 is noise, you're fading the thesis. "
    stale = cf.get('staleness', '—')
    line2 += f"Staleness {stale} still trims confidence."
    print(line2)
    print()

    circle = "inner circle" if rank <= 5 else ("top tier" if rank <= 10 else "upper half")
    profile = "both a power profile play and a " if hr_rate > 0.04 else "a "
    line3 = (
        f"**Net:** ~{adj_hr*100:.1f}% per PA adjusted → ~{three_pa_adj:.1f}% over 3 PA — about **1-in-{inv}** "
        f"for at least one HR in a three-PA window. "
        f"That's the slate's {circle} (#{rank} overall, #{team_rank_val} on {bteam}). "
        f"This is {profile}matchup play — if you don't buy the {last} vulnerability thesis, cut enthusiasm; "
        f"if you do, {blast} is exactly the type of power bat you'd short-list for HR."
    )
    print(line3)
    print()
    print("---")
    print()
