"""
Generate per-player outcome probability table for all qualifying 2026 batters.
Uses the league-wide trained model to produce:
  P(Hit), P(HR), P(2B), P(XBH), P(2+ TB), P(BB), P(K)

Output: data/reports/league_predictions.csv  +  prints to stdout
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import pickle
import numpy as np
import pandas as pd
from config import RAW_DIR, MASTER_DIR, REPORTS_DIR


def load_model():
    model_path = MASTER_DIR / "models" / "best_model.pkl"
    with open(model_path, "rb") as f:
        return pickle.load(f)


def get_recent_features_for_batters(batter_ids: set) -> pd.DataFrame:
    """Get recent rolling features for each qualifying batter from 2025 val data."""
    val_path = MASTER_DIR / "features_val_league.parquet"
    val = pd.read_parquet(val_path)

    feat_cols = [c for c in val.columns if c.startswith(("roll_", "cum_", "vs_lhp", "month",
                                                          "day_of_week", "pitch_count", "in_zone"))]

    relevant = val[val["batter"].isin(batter_ids)].copy()
    if relevant.empty:
        return pd.DataFrame()

    recent = relevant.sort_values("game_date").groupby("batter").tail(1)
    return recent


def compute_outcome_rates(val_data: pd.DataFrame, batter_ids: set) -> pd.DataFrame:
    """Compute empirical outcome rates from the validation (2025) data."""
    pa_path = RAW_DIR / "statcast_pa_level_league.parquet"
    pa = pd.read_parquet(pa_path)
    pa_2025 = pa[pa["game_year"] == 2025].copy()

    HIT_2B = {"double"}
    HIT_XBH = {"double", "triple", "home_run"}

    pa_2025["is_2b"] = pa_2025["events"].isin(HIT_2B).astype(int)
    pa_2025["is_xbh"] = pa_2025["events"].isin(HIT_XBH).astype(int)

    tb_map = {"single": 1, "double": 2, "triple": 3, "home_run": 4}
    pa_2025["tb"] = pa_2025["events"].map(tb_map).fillna(0).astype(int)
    pa_2025["tb_2plus"] = (pa_2025["tb"] >= 2).astype(int)

    stats = pa_2025.groupby("batter").agg(
        pa_count=("events", "count"),
        hit_rate=("is_hit", "mean"),
        hr_rate=("is_hr", "mean"),
        db_rate=("is_2b", "mean"),
        xbh_rate=("is_xbh", "mean"),
        tb2_rate=("tb_2plus", "mean"),
        bb_rate=("is_walk", "mean"),
        k_rate=("is_strikeout", "mean"),
    ).reset_index()

    return stats


def simulate_game_probs(p_hit, p_hr_given_pa, pa_per_game=4, n_sims=50000,
                        p_k=0.222, p_bb=0.084):
    """Multi-outcome Monte Carlo for per-game hit/HR probability.

    Uses multinomial draws (Hit / K / BB / Other Out) so that high-K batters
    are properly penalized for multi-hit upside.
    """
    total = p_hit + p_k + p_bb
    if total > 1.0:
        scale = 1.0 / total
        p_hit, p_k, p_bb = p_hit * scale, p_k * scale, p_bb * scale
    p_other = 1.0 - p_hit - p_k - p_bb

    rng = np.random.default_rng(42)
    outcomes = rng.multinomial(pa_per_game, [p_hit, p_k, p_bb, p_other], size=n_sims)
    hits = outcomes[:, 0]
    hrs = rng.binomial(pa_per_game, p_hr_given_pa, n_sims)
    return {
        "p_any_hit": float((hits >= 1).mean()),
        "p_multi_hit": float((hits >= 2).mean()),
        "p_any_hr": float((hrs >= 1).mean()),
    }


def main():
    print("=" * 70)
    print("  LEAGUE-WIDE PREDICTION TABLE (2026 Qualifying Batters)")
    print("=" * 70)

    qualifying = pd.read_csv(RAW_DIR / "qualifying_batters_2026.csv")
    batter_ids = set(qualifying["mlbam_id"].values)

    model = load_model()
    print(f"Model loaded: {type(model).__name__}")

    feat_cols = sorted([
        'cum_career_ba_vs_lhp', 'cum_career_ba_vs_rhp', 'cum_career_hr_vs_lhp',
        'cum_career_hr_vs_rhp', 'cum_career_ops_vs_lhp', 'cum_career_ops_vs_rhp',
        'cum_career_pa_vs_lhp', 'cum_career_pa_vs_rhp', 'day_of_week', 'in_zone',
        'month', 'pitch_count', 'roll_ba_10', 'roll_ba_100', 'roll_ba_30',
        'roll_barrel_10', 'roll_barrel_100', 'roll_barrel_30', 'roll_bb_rate_10',
        'roll_bb_rate_100', 'roll_bb_rate_30', 'roll_ev_10', 'roll_ev_100',
        'roll_ev_30', 'roll_hr_rate_10', 'roll_hr_rate_100', 'roll_hr_rate_30',
        'roll_k_rate_10', 'roll_k_rate_100', 'roll_k_rate_30', 'roll_la_10',
        'roll_la_100', 'roll_la_30', 'vs_lhp',
    ])

    val = pd.read_parquet(MASTER_DIR / "features_val_league.parquet")

    relevant = val[val["batter"].isin(batter_ids)].copy()
    print(f"  {relevant['batter'].nunique()} qualifying batters found in 2025 validation data")

    recent = relevant.sort_values(["batter", "game_date"]).groupby("batter").tail(50)

    model_preds = []
    for bid, grp in recent.groupby("batter"):
        X = grp[feat_cols].fillna(0)
        probs = model.predict_proba(X)[:, 1]
        model_preds.append({"batter": bid, "p_hit_model": probs.mean()})
    model_preds = pd.DataFrame(model_preds)

    empirical = compute_outcome_rates(val, batter_ids)

    result = qualifying[["mlbam_id", "name", "team", "pos", "AB", "PA"]].merge(
        model_preds, left_on="mlbam_id", right_on="batter", how="left"
    ).merge(
        empirical, left_on="mlbam_id", right_on="batter", how="left"
    )

    game_probs = []
    for _, row in result.iterrows():
        p_hit = row.get("p_hit_model", row.get("hit_rate", 0.22))
        p_hr = row.get("hr_rate", 0.03)
        if pd.isna(p_hit):
            p_hit = 0.22
        if pd.isna(p_hr):
            p_hr = 0.03
        gp = simulate_game_probs(p_hit, p_hr)
        game_probs.append(gp)

    gp_df = pd.DataFrame(game_probs)
    result = pd.concat([result.reset_index(drop=True), gp_df], axis=1)

    result = result.rename(columns={
        "p_hit_model": "P(Hit/PA)",
        "hr_rate": "P(HR/PA)",
        "db_rate": "P(2B/PA)",
        "xbh_rate": "P(XBH/PA)",
        "tb2_rate": "P(2+TB/PA)",
        "bb_rate": "P(BB/PA)",
        "k_rate": "P(K/PA)",
        "p_any_hit": "P(Hit/Game)",
        "p_multi_hit": "P(2+Hits/Game)",
        "p_any_hr": "P(HR/Game)",
    })

    display_cols = ["name", "team", "pos", "AB",
                    "P(Hit/PA)", "P(HR/PA)", "P(2B/PA)", "P(XBH/PA)",
                    "P(2+TB/PA)", "P(BB/PA)", "P(K/PA)",
                    "P(Hit/Game)", "P(2+Hits/Game)", "P(HR/Game)"]
    available = [c for c in display_cols if c in result.columns]

    final = result[available].copy()
    final = final.sort_values("P(Hit/PA)", ascending=False).reset_index(drop=True)

    for col in final.columns:
        if col.startswith("P("):
            final[col] = final[col].apply(lambda x: f"{x:.1%}" if pd.notna(x) else "N/A")

    out_path = REPORTS_DIR / "league_predictions.csv"
    final.to_csv(out_path, index=False)

    print(f"\n{'='*70}")
    print(f"  ALL {len(final)} QUALIFYING BATTERS — OUTCOME PROBABILITIES")
    print(f"{'='*70}\n")
    pd.set_option("display.max_rows", 500)
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 20)
    print(final.to_string(index=False))
    print(f"\nSaved → {out_path}")


if __name__ == "__main__":
    main()
