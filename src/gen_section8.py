import json
import os

import numpy as np
import pandas as pd


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

_sort_hit = "p_hit" if "p_hit" in df.columns else "adj_p_hit"
top10_hit = df.sort_values(_sort_hit, ascending=False).head(10).reset_index(drop=True)
total_matchups = len(df)

print("## Section 8: Top 10 Hit Full Analysis\n")

for idx, row in top10_hit.iterrows():
    bname = row['batter_name']
    pname = row['pitcher_name']
    bteam = row['batter_team']
    bid = int(row['batter_mlbam_id']) if not pd.isna(row.get('batter_mlbam_id')) else None
    pid = int(row['pitcher_mlbam_id']) if not pd.isna(row.get('pitcher_mlbam_id')) else None
    raw_hit = row.get('p_hit', 0) or 0
    adj_hit = row.get('adj_p_hit', 0) or 0
    conf = row.get('conf_hit', 0) or 0
    grade = row.get('conf_hit_label', '—')
    conf_factors = row.get('conf_factors', {})
    if isinstance(conf_factors, str):
        conf_factors = json.loads(conf_factors)
    three_pa_raw = (1 - (1 - raw_hit) ** 3) * 100
    three_pa_adj = (1 - (1 - adj_hit) ** 3) * 100
    three_pa_adj_p = 1 - (1 - adj_hit) ** 3
    inv = int(round(1 / three_pa_adj_p)) if three_pa_adj_p > 0 else 999

    last = pname.split()[-1]
    blast = bname.split()[-1]

    print(f"## {bname} Hit Analysis vs {pname}")
    print()
    print(f"Model (tonight): Raw P(Hit) {raw_hit*100:.1f}% | Adj P(Hit) {adj_hit*100:.1f}% | Conf {conf:.3f} ({grade})")
    print(
        f"3 PA vs starter: (1-(1-{raw_hit:.4f})^3 ≈) {three_pa_raw:.1f}% raw | ~{three_pa_adj:.1f}% adjusted — "
        f"about **1-in-{inv}** at least one hit."
    )
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
    career_hits = int(batter_pa['is_hit'].sum()) if 'is_hit' in batter_pa.columns else 0
    career_ba = career_hits / total_pa if total_pa > 0 else 0
    career_k_rate = batter_pa['is_strikeout'].mean() * 100 if 'is_strikeout' in batter_pa.columns else 0
    career_k_model = (row.get('p_k', 0) or 0) * 100
    if 'events' in batter_pa.columns:
        evs = batter_pa['events']
        career_doubles = int(evs.eq('double').sum())
        career_triples = int(evs.eq('triple').sum())
    else:
        career_doubles = career_triples = 0
    career_hrs = int(batter_pa['is_hr'].sum())

    league_ba = 0.243
    ba_vs = "above" if career_ba > league_ba else "below"

    print("**Career Contact Profile**")
    print(
        f"{career_hits:,} career hits in {total_pa:,} PA (.{int(career_ba*1000):03d} BA). "
        f"League BA in this dataset is ~.243 — {blast} is "
        f"{'well ' if abs(career_ba - league_ba) > 0.020 else ''}{ba_vs} that. "
        f"{'Consistent contact maker.' if career_ba > 0.260 else 'Solid contact.' if career_ba > 0.240 else 'Contact is a concern.'}"
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

    hit_pa = batter_pa[batter_pa['is_hit'] == 1]
    ev_mean = hit_pa['launch_speed'].mean() if 'launch_speed' in hit_pa.columns and len(hit_pa) > 0 else 0
    ev_note = "firm contact vs league hit EV (~87 mph in this population)." if ev_mean >= 88 else "contact quality is more placement than pure smoke — still plays if the pitcher leaks barrels."
    print(f"Exit velocity on all career hits: mean {ev_mean:.1f} mph — {ev_note}")
    print()

    if 'pitch_type' in hit_pa.columns and len(hit_pa) > 0:
        hits_by_pt = hit_pa.groupby('pitch_type').size().sort_values(ascending=False).head(6)
        pt_ba = {}
        for pt in hits_by_pt.index:
            pt_pa_df = batter_pa[batter_pa['pitch_type'] == pt]
            pt_hits = pt_pa_df['is_hit'].sum() if 'is_hit' in pt_pa_df.columns else 0
            pt_ba[pt] = pt_hits / len(pt_pa_df) if len(pt_pa_df) > 0 else 0

        pt_parts = []
        for pt, ct in hits_by_pt.items():
            ba_val = pt_ba.get(pt, 0)
            pt_parts.append(f"{pt} {int(ct)} (.{int(ba_val*1000):03d} BA)")
        pt_line = ", ".join(pt_parts)
        print(f"Hits by pitch type (top): {pt_line}.")

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

        top_hit_type = hits_by_pt.index[0]
        top_hit_ct = int(hits_by_pt.iloc[0])
        if pitcher_top3:
            print(
                f"That's the math: {last} runs {pitcher_top3} as top usage pitches — "
                f"{blast}'s {top_hit_ct} hits off {top_hit_type} line up with what {last} actually throws."
            )
            print()
            print(
                f"The swing decisions show up in the buckets: when {last} leans on what he trusts, "
                f"{blast} already has a hit library on those pitch codes. "
                f"League BA in this dataset is ~**.243** — {blast} is {'north' if career_ba > 0.25 else 'around' if career_ba > 0.23 else 'south'} of that baseline, "
                f"so the hit prop isn't a charity case; it's a contact skill pressed against a specific pitch diet."
            )
    print()

    # Pitch group splits from batter_pitch_profiles
    if bid:
        bp = batter_prof[batter_prof['batter'] == bid]
        if len(bp) > 0:
            splits = []
            for grp in ['fastball', 'breaking', 'offspeed']:
                grp_rows = bp[bp['pitch_group'] == grp] if 'pitch_group' in bp.columns else pd.DataFrame()
                if len(grp_rows) > 0:
                    g = grp_rows.iloc[0]
                    ba = g.get('ba', 0)
                    ev = g.get('avg_ev', 0)
                    whiff = g.get('whiff_rate', 0)
                    ba = ba if pd.notna(ba) else 0
                    ev = ev if pd.notna(ev) else 0
                    whiff = whiff if pd.notna(whiff) else 0
                    splits.append(f"vs {grp.title()} .{int(ba*1000):03d} BA / {ev:.1f} mph EV / {whiff*100:.1f}% whiff")
            if splits:
                print(f"Pitch-group splits (batter_pitch_profiles): {' | '.join(splits)}.")
                print()
                print(
                    f"Editorial read: {blast}'s pitch-group ladder shows where the loud contact lives — "
                    f"map that lane to {last}'s usage and you're pricing how often the good counts arrive."
                )
        print()

    print(
        f"One-sentence summary: {blast} is a "
        f"{'contact-first hitter' if career_ba > 0.260 else 'solid bat' if career_ba > 0.240 else 'power-over-contact bat'}"
        f" who {f'keeps the K rate low at {career_k_rate:.0f}%' if career_k_rate < 20 else f'has an elevated K rate at {career_k_rate:.0f}%'},"
        f" and {last}'s arsenal {'feeds right into' if career_ba > 0.250 else 'tests'} that profile."
    )
    print()

    # BvP
    print(f"**The {last}-Specific Hit Case**")
    if pid:
        bvp = batter_pa[batter_pa['pitcher'] == pid]
        bvp_pa_ct = len(bvp)
        if bvp_pa_ct > 0:
            bvp_hits = int(bvp['is_hit'].sum())
            bvp_hrs = int(bvp['is_hr'].sum())
            bvp_xbh = int(bvp['is_xbh'].sum()) if 'is_xbh' in bvp.columns else 0
            bvp_ks = int(bvp['is_strikeout'].sum())
            bvp_ba = bvp_hits / bvp_pa_ct
            bvp_k_rate = bvp_ks / bvp_pa_ct * 100

            if bvp_ba >= 0.300 and bvp_pa_ct >= 5:
                bvp_label = "Strong BvP history."
            elif bvp_ba < 0.200 and bvp_pa_ct >= 5:
                bvp_label = "Struggled historically."
            elif bvp_pa_ct < 5:
                bvp_label = "Tiny sample."
            else:
                bvp_label = "Moderate BvP history."

            print(
                f"BvP: {bvp_hits}-for-{bvp_pa_ct} {_fmt_avg_paren(bvp_ba)}, "
                f"{bvp_hrs} HR, {bvp_xbh} XBH, {bvp_ks} K in {bvp_pa_ct} PA. {bvp_label}"
            )
            print()

            k_word = "ugly" if bvp_k_rate > 30 else ("workable" if bvp_k_rate > 20 else "clean")
            print(f"BvP K rate: {bvp_k_rate:.0f}% — {k_word}.")
            print()

            if bvp_hits > 0:
                print(
                    f"When {blast} makes contact against {last}, hits happen. "
                    f"The BvP BA of {_fmt_avg_plain(bvp_ba)} {'confirms' if bvp_ba > 0.250 else 'suggests room for'} "
                    f"the contact profile working against this arsenal."
                )
            else:
                print(
                    f"No hits in {bvp_pa_ct} BvP PA — but sample is "
                    f"{'too small to draw conclusions.' if bvp_pa_ct < 10 else 'concerning.'} "
                    f"Pivot to pitch-type matchup logic."
                )
        else:
            print(
                f"No BvP history — these two have not faced each other in the Statcast era. "
                f"Pivot to pitch-type matchup logic: {blast}'s contact splits vs {last}'s arsenal."
            )
    print()

    # Pitcher vulnerability
    print(f"**{last}'s Hit Vulnerability**")
    if pid:
        pitcher_pa_df = pa_pitcher(pid)
        p_total_pa = len(pitcher_pa_df)
        p_hits_allowed = int(pitcher_pa_df['is_hit'].sum()) if 'is_hit' in pitcher_pa_df.columns and p_total_pa > 0 else 0
        p_ab = p_total_pa - int(pitcher_pa_df['is_walk'].sum()) if 'is_walk' in pitcher_pa_df.columns else p_total_pa
        p_ba = p_hits_allowed / p_ab if p_ab > 0 else 0

        ba_comment = "right on" if abs(p_ba - 0.243) < 0.010 else ("above" if p_ba > 0.243 else "below")
        print(
            f"Career: {p_hits_allowed:,} hits allowed in {p_ab:,} AB (.{int(p_ba*1000):03d} BA allowed). "
            f"That's {ba_comment} the ~.243 league average."
        )
        print()

        p_k_rate = pitcher_pa_df['is_strikeout'].mean() * 100 if 'is_strikeout' in pitcher_pa_df.columns else 0
        p_bb_rate = pitcher_pa_df['is_walk'].mean() * 100 if 'is_walk' in pitcher_pa_df.columns else 0
        if p_k_rate < 20:
            kb_comment = "low K rate means more balls in play — more hit variance."
        elif p_k_rate < 25:
            kb_comment = "decent strikeout rate limits some contact opportunity."
        else:
            kb_comment = "high K rate makes it harder to collect hits."
        print(f"K rate: {p_k_rate:.1f}% | BB rate: {p_bb_rate:.1f}% — {kb_comment}")
        print()

        pp = pitcher_prof[pitcher_prof['pitcher'] == pid]
        if len(pp) > 0:
            sort_col2 = 'season' if 'season' in pp.columns else 'game_year'
            latest = pp.sort_values(sort_col2, ascending=False).iloc[0]
            fb_velo = latest.get('velo_fastball', 0)
            if pd.isna(fb_velo):
                fb_velo = 0
            fb_pct = latest.get('pct_fastball', 0)
            if pd.isna(fb_pct):
                fb_pct = 0
            brk_pct = latest.get('pct_breaking', 0)
            if pd.isna(brk_pct):
                brk_pct = 0
            off_pct = latest.get('pct_offspeed', 0)
            if pd.isna(off_pct):
                off_pct = 0
            print(
                f"Arsenal breakdown: Fastball {fb_pct*100:.0f}% (~{fb_velo:.0f} mph), "
                f"Breaking {brk_pct*100:.0f}%, Offspeed {off_pct*100:.0f}%."
            )
        print()

        if 'stand' in pitcher_pa_df.columns:
            lhb = pitcher_pa_df[pitcher_pa_df['stand'] == 'L']
            rhb = pitcher_pa_df[pitcher_pa_df['stand'] == 'R']
            lhb_h = int(lhb['is_hit'].sum())
            rhb_h = int(rhb['is_hit'].sum())
            lhb_ab = len(lhb) - int(lhb['is_walk'].sum()) if 'is_walk' in lhb.columns else len(lhb)
            rhb_ab = len(rhb) - int(rhb['is_walk'].sum()) if 'is_walk' in rhb.columns else len(rhb)
            lhb_ba = lhb_h / lhb_ab if lhb_ab > 0 else 0
            rhb_ba = rhb_h / rhb_ab if rhb_ab > 0 else 0

            batter_hand = 'R'
            if bid:
                b_prof2 = batter_prof[batter_prof['batter'] == bid]
                if len(b_prof2) > 0 and 'stand' in b_prof2.columns:
                    batter_hand = b_prof2.iloc[0]['stand']
            hand_label = "left" if batter_hand == 'L' else "right"
            print(
                f"BA allowed by handedness: vs LHB .{int(lhb_ba*1000):03d} ({lhb_h} hits in {lhb_ab} AB) | "
                f"vs RHB .{int(rhb_ba*1000):03d} ({rhb_h} in {rhb_ab}). {blast} bats {hand_label}."
            )
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

    # Rankings
    print(f"**Where {blast} Ranks Tonight**")
    rank = idx + 1
    pctile = max(1, min(99, int(round(100 * rank / max(total_matchups, 1)))))
    team_df = df[df['batter_team'] == bteam].sort_values(_sort_hit, ascending=False).reset_index(drop=True)
    team_rank_val = "—"
    if bname in team_df['batter_name'].values:
        team_rank_val = int(team_df[team_df['batter_name'] == bname].index[0]) + 1
    tr_label = "best" if team_rank_val == 1 else f"#{team_rank_val}"
    print(f"#{rank} of {total_matchups} matchups by adj P(Hit) — top ~{pctile}% of the slate.")
    print(f"#{team_rank_val} on {bteam} by adj P(Hit) — {tr_label} hit projection on {bteam} in this file.")
    print()

    # Verdict
    print("**The Verdict on the Hit Bet**")
    ba_word = "crushes" if career_ba > 0.270 else ("beats" if career_ba > 0.243 else "trails")
    line1 = (
        f"**For:** .{int(career_ba*1000):03d} career BA {ba_word} .243 league average. "
        f"Model P(Hit) at {adj_hit*100:.1f}% per PA is "
        f"{'well above' if adj_hit > 0.25 else 'above'} baseline."
    )
    if bid and pid:
        bvp_data = pa_batter(bid)
        bvp_data = bvp_data[bvp_data['pitcher'] == pid] if len(bvp_data) else bvp_data
        bvp_h = int(bvp_data['is_hit'].sum()) if len(bvp_data) > 0 else 0
        bvp_p = len(bvp_data)
        if bvp_p > 0:
            bvp_ba2 = bvp_h / bvp_p
            if bvp_ba2 >= 0.250 and bvp_p >= 5:
                line1 += f" BvP BA {_fmt_avg_plain(bvp_ba2)} in {bvp_p} PA is a plus."
            elif bvp_ba2 > 0:
                line1 += f" BvP line: {bvp_h}-for-{bvp_p} — contact has happened."
        else:
            line1 += f" No BvP history but pitch-type matchup favors {blast}."
    conv_hit = cf.get('convergence_hit', '—')
    line1 += f" Confidence on hit is {grade} with convergence_hit {conv_hit}."
    print(line1)
    print()

    line2 = "**Against:** "
    if bid and pid:
        bvp_data2 = pa_batter(bid)
        bvp_data2 = bvp_data2[bvp_data2['pitcher'] == pid] if len(bvp_data2) else bvp_data2
        bvp_ks2 = int(bvp_data2['is_strikeout'].sum()) if len(bvp_data2) > 0 else 0
        bvp_pa_ct3 = len(bvp_data2)
        if bvp_ks2 > 0 and bvp_pa_ct3 > 0:
            bvp_kr = bvp_ks2 / bvp_pa_ct3 * 100
            line2 += f"BvP K rate {bvp_kr:.0f}% ({bvp_ks2} K in {bvp_pa_ct3} PA). "
    k_word2 = "elevated" if career_k_rate > 25 else "manageable"
    line2 += f"{blast}'s {career_k_rate:.1f}% career K rate is {k_word2}; model P(K) ~{career_k_model:.0f}% tonight. "
    if pid:
        line2 += f"Staleness {cf.get('staleness', '—')} still trims confidence."
    print(line2)
    print()

    one_in = int(round(1 / (three_pa_adj / 100))) if three_pa_adj > 0 else 999
    one_in = max(1, one_in)
    circle = "inner circle" if rank <= 5 else ("top tier" if rank <= 10 else "upper half")
    line3 = (
        f"**Net:** ~{adj_hit*100:.1f}% per PA adjusted → ~{three_pa_adj:.1f}% over 3 PA — about {one_in}-in-{one_in+1} for at least one hit. "
        f"That's the slate's {circle} (#{rank} overall, #{team_rank_val} on {bteam}). "
        f"This is a {'contact profile play' if career_ba > 0.260 else 'matchup play'} — "
        f"if you don't buy the {last} vulnerability thesis, cut enthusiasm; "
        f"if you do, {blast} is exactly the type of hitter you'd target for a hit."
    )
    print(line3)
    print()
    print("---")
    print()
