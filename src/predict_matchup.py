"""
CLI tool: predict a specific batter vs pitcher matchup.

Usage:
  python src/predict_matchup.py "Shohei Ohtani" "David Peterson"
"""
import argparse
import json
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

from narrative_engine import (load_models, load_feature_cols, load_pitcher_profiles,
                               load_batter_pitch_profiles, load_bvp_history,
                               build_feature_vector, generate_narrative,
                               resolve_pitcher_throws,
                               apply_vs_hand_and_bvp_posteriors,
                               _enforce_hit_xbh_hr_order,
                               LEAGUE_AVG, TARGETS, pct)
import pandas as pd
from config import (
    EXPERIMENT_MODEL_DIR,
    EXPERIMENT_VAL_FEATURES,
    MASTER_DIR,
    RAW_DIR,
    RECENCY_MODEL_DIR,
    RECENCY_VAL_FEATURES,
)


def find_player_id(name_query: str, players_df: pd.DataFrame, col="player_name_clean") -> int | None:
    q = name_query.lower().strip()
    parts = q.split()
    reversed_q = f"{parts[-1]}, {' '.join(parts[:-1])}" if len(parts) >= 2 else q

    for _, row in players_df.iterrows():
        val = str(row.get(col, "")).lower().strip()
        if val == q or val == reversed_q:
            return int(row["batter"])
    for _, row in players_df.iterrows():
        val = str(row.get(col, "")).lower().strip()
        if q in val or reversed_q in val:
            return int(row["batter"])
    return None


def find_pitcher_id(name_query: str, profiles: pd.DataFrame) -> int | None:
    pitchers_csv = pd.read_csv(RAW_DIR / "qualifying_pitchers_2026.csv")
    q = name_query.lower().strip()
    for _, row in pitchers_csv.iterrows():
        if str(row.get("name", "")).lower().strip() == q:
            return int(row["mlbam_id"])
    for _, row in pitchers_csv.iterrows():
        if q in str(row.get("name", "")).lower():
            return int(row["mlbam_id"])
    return None


def main():
    parser = argparse.ArgumentParser(description="Predict batter vs pitcher matchup")
    parser.add_argument("batter", help="Batter name (e.g. 'Shohei Ohtani')")
    parser.add_argument("pitcher", help="Pitcher name (e.g. 'David Peterson')")
    parser.add_argument(
        "--experiment",
        action="store_true",
        help="Use exp_bpt_xwoba models + features_val_league_exp.parquet",
    )
    parser.add_argument(
        "--recency",
        action="store_true",
        help="Use exp_recency_l3l5 models + features_val_league_recency.parquet",
    )
    parser.add_argument(
        "--pitcher-throws",
        choices=["L", "R", "l", "r"],
        default=None,
        help="Override pitcher handedness (L/R) for platoon-aligned features tonight",
    )
    args = parser.parse_args()

    if args.experiment and args.recency:
        print("ERROR: Use only one of --experiment or --recency")
        return

    if args.recency:
        model_dir = RECENCY_MODEL_DIR
        val_path = RECENCY_VAL_FEATURES
    elif args.experiment:
        model_dir = EXPERIMENT_MODEL_DIR
        val_path = EXPERIMENT_VAL_FEATURES
    else:
        model_dir = None
        val_path = MASTER_DIR / "features_val_league.parquet"

    if (args.experiment or args.recency) and not val_path.exists():
        print(f"ERROR: Val features not found: {val_path}")
        return
    if args.recency and not (RECENCY_MODEL_DIR / "feature_columns.json").exists():
        print(f"ERROR: Recency model bundle not found under {RECENCY_MODEL_DIR}")
        return

    label = "recency" if args.recency else ("experiment" if args.experiment else "production")
    print(f"Loading models and data... ({label})")
    models = load_models(model_dir)
    feat_cols = load_feature_cols(model_dir)
    pitcher_profiles = load_pitcher_profiles()
    batter_profiles = load_batter_pitch_profiles()
    bvp_history = load_bvp_history()
    val_df = pd.read_parquet(val_path)

    pa_data = pd.read_parquet(RAW_DIR / "statcast_pa_level_league.parquet")
    hit_events = {"single", "double", "triple", "home_run"}
    pa_data["is_hit_flag"] = pa_data["events"].isin(hit_events).astype(int)
    pa_data["is_hr_flag"] = (pa_data["events"] == "home_run").astype(int)
    pa_data["is_k_flag"] = pa_data["events"].str.contains("strikeout", case=False, na=False).astype(int)
    pa_data["is_bb_flag"] = pa_data["events"].isin({"walk", "intent_walk"}).astype(int)
    pa_data["is_xbh_flag"] = pa_data["events"].isin({"double", "triple", "home_run"}).astype(int)

    batters_csv = pd.read_csv(RAW_DIR / "qualifying_batters_2026.csv")
    q = args.batter.lower().strip()
    match = batters_csv[batters_csv["name"].str.lower() == q]
    if match.empty:
        match = batters_csv[batters_csv["name"].str.lower().str.contains(q)]
    batter_id = int(match.iloc[0]["mlbam_id"]) if not match.empty else None
    if batter_id is None:
        print(f"ERROR: Could not find batter '{args.batter}'")
        return

    pitcher_id = find_pitcher_id(args.pitcher, pitcher_profiles)
    if pitcher_id is None:
        print(f"ERROR: Could not find pitcher '{args.pitcher}'")
        return

    print(f"  Batter: {args.batter} (ID {batter_id})")
    print(f"  Pitcher: {args.pitcher} (ID {pitcher_id})")

    throws_hint = None
    if args.pitcher_throws:
        throws_hint = str(args.pitcher_throws).strip().upper()[0]

    if "game_date" in val_df.columns:
        _gd = val_df["game_date"].max()
        game_ctx = {"game_date": _gd, "game_year": int(pd.Timestamp(_gd).year)}
    else:
        game_ctx = {}
    X, extra_platoon = build_feature_vector(
        batter_id,
        pitcher_id,
        feat_cols,
        val_df,
        pitcher_profiles,
        batter_profiles,
        pa_data=pa_data,
        skip_rolling_dampen=args.recency,
        pitcher_throws_hint=throws_hint,
        game_context=game_ctx,
    )

    throws_resolved = resolve_pitcher_throws(pitcher_id, pitcher_profiles, throws_hint)

    pp = pitcher_profiles[pitcher_profiles["pitcher"] == pitcher_id]
    pitcher_prof = pp.sort_values("season").iloc[-1].to_dict() if not pp.empty else {}

    bpp_rows = batter_profiles[batter_profiles["batter"] == batter_id]
    bpp = bpp_rows.sort_values("season").iloc[-1].to_dict() if not bpp_rows.empty else {}

    bvp_match = bvp_history[(bvp_history["batter"] == batter_id) &
                            (bvp_history["pitcher"] == pitcher_id)]
    bvp_row = bvp_match.iloc[0].to_dict() if not bvp_match.empty else None

    if X is None:
        print("WARNING: No feature data for this batter, using league averages")
        predictions = {t: LEAGUE_AVG.get(t.replace("is_", ""), 0.2) for t in TARGETS}
        feat_snap = None
        extra_platoon = {}
        plat_raw_pen = False
    else:
        predictions = {}
        for target, model in models.items():
            predictions[target] = float(model.predict_proba(X)[:, 1][0])
        feat_snap = X.iloc[0].to_dict()
        feat_snap.update(extra_platoon or {})
        predictions, _ = apply_vs_hand_and_bvp_posteriors(
            predictions, feat_snap, throws_resolved, extra_platoon, bvp_row
        )
        plat_raw_pen = False

    _enforce_hit_xbh_hr_order(predictions)

    batter_pas = pa_data[pa_data["batter"] == batter_id]
    career_rates = {
        "hit": batter_pas["is_hit_flag"].mean() if len(batter_pas) > 0 else LEAGUE_AVG["hit"],
        "hr": batter_pas["is_hr_flag"].mean() if len(batter_pas) > 0 else LEAGUE_AVG["hr"],
        "xbh": batter_pas["is_xbh_flag"].mean() if len(batter_pas) > 0 else LEAGUE_AVG["xbh"],
        "strikeout": batter_pas["is_k_flag"].mean() if len(batter_pas) > 0 else LEAGUE_AVG["strikeout"],
        "walk": batter_pas["is_bb_flag"].mean() if len(batter_pas) > 0 else LEAGUE_AVG["walk"],
    }

    pitchers_csv = pd.read_csv(RAW_DIR / "qualifying_pitchers_2026.csv")
    pitcher_row = pitchers_csv[pitchers_csv["mlbam_id"] == pitcher_id]
    pitcher_team = pitcher_row.iloc[0]["team"] if not pitcher_row.empty else "?"

    result = generate_narrative(
        args.batter, "?", args.pitcher, pitcher_team,
        predictions, pitcher_prof, bpp, bvp_row, career_rates,
        feature_snapshot=feat_snap,
        pitcher_throws=throws_resolved,
        platoon_raw_shrink_applied=plat_raw_pen,
    )

    throws = result.get("pitcher_throws", "R")

    print(f"\n{'='*60}")
    print(f"  {args.batter} vs {args.pitcher} ({pitcher_team}, {'LHP' if throws == 'L' else 'RHP'})")
    print(f"  Tier: {result['tier']}")
    print(f"{'='*60}")
    print(f"  P(Hit):       {pct(result['p_hit']):>7s}  | Career: {pct(result['career_hit'])} | Lg: {pct(0.222)}")
    print(f"  P(HR):        {pct(result['p_hr']):>7s}  | Career: {pct(result['career_hr'])} | Lg: {pct(0.031)}")
    print(f"  P(K):         {pct(result['p_k']):>7s}  | Career: {pct(result['career_k'])} | Lg: {pct(0.222)}")
    print(f"  P(BB):        {pct(result['p_bb']):>7s}  | Career: {pct(result['career_bb'])} | Lg: {pct(0.084)}")
    print(f"  P(XBH):       {pct(result['p_xbh']):>7s}  | Lg: {pct(0.079)}")
    print(f"  P(Multi-Hit): {pct(result['p_multi_hit']):>7s}  | Sim: 10K games, 3 PAs")
    print(f"  P(Any Hit):   {pct(result['p_any_hit']):>7s}")
    print(f"\n  Hit: {result['hit_narrative']}")
    print(f"  K:   {result['k_narrative']}")
    if result['bvp_text']:
        print(f"  BvP: {result['bvp_text']}")


if __name__ == "__main__":
    main()
