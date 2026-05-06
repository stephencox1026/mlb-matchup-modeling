"""
Game simulation: convert PA-level hit probabilities into per-game distributions.

For each of the 14 Dodgers hitters, simulate their expected performance in a
single game against a left-handed pitcher using the trained ML model's
calibrated probabilities + career split context.

Output:
  data/reports/game_sim_results.csv
  docs/figures/game_sim_distributions.png
"""
import pickle
import numpy as np
import pandas as pd
from config import MASTER_DIR, REPORTS_DIR, FIGURES_DIR, RAW_DIR

MODEL_PATH = MASTER_DIR / "models" / "best_model.pkl"
VAL_PATH = MASTER_DIR / "features_val.parquet"
HIST_SPLITS_PATH = RAW_DIR / "historical_splits.parquet"
N_SIMS = 10_000
TYPICAL_PA_PER_GAME = 4


def load_model():
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)


def get_player_probs(model, val_df: pd.DataFrame) -> pd.DataFrame:
    """Get mean predicted P(hit) per player from validation set, split by pitcher hand."""
    feat_cols = sorted([c for c in val_df.columns if c.startswith((
        "roll_", "cum_", "vs_lhp", "month", "day_of_week", "pitch_count", "in_zone"
    ))])

    val_df = val_df.copy()
    val_df["p_hit"] = model.predict_proba(val_df[feat_cols])[:, 1]

    vs_lhp = val_df[val_df.get("vs_lhp", pd.Series(dtype=int)) == 1] if "vs_lhp" in val_df.columns else pd.DataFrame()

    player_probs = []
    for name, grp in val_df.groupby("player_name_clean"):
        row = {
            "name": name,
            "n_pa_val": len(grp),
            "p_hit_overall": grp["p_hit"].mean(),
            "actual_hit_rate": grp["is_hit"].mean(),
        }
        lhp_grp = grp[grp["vs_lhp"] == 1] if "vs_lhp" in grp.columns else pd.DataFrame()
        if len(lhp_grp) > 0:
            row["p_hit_vs_lhp"] = lhp_grp["p_hit"].mean()
            row["n_pa_vs_lhp"] = len(lhp_grp)
        else:
            row["p_hit_vs_lhp"] = grp["p_hit"].mean()
            row["n_pa_vs_lhp"] = 0

        player_probs.append(row)

    return pd.DataFrame(player_probs)


def simulate_game(p_hit: float, p_hr_given_hit: float = 0.10,
                  pa_per_game: int = TYPICAL_PA_PER_GAME,
                  n_sims: int = N_SIMS) -> dict:
    """
    Simulate n_sims games for a batter.
    Each PA: coin flip with P(hit) = p_hit.
    If hit, coin flip for HR with P(HR|hit) = p_hr_given_hit.
    Non-HR hits: uniformly distributed as 1B (TB=1), 2B (TB=2), 3B (TB=3).
    """
    rng = np.random.default_rng(42)
    hits = rng.binomial(pa_per_game, p_hit, n_sims)
    hrs = np.array([rng.binomial(h, p_hr_given_hit) for h in hits])

    tb = np.zeros(n_sims)
    for i in range(n_sims):
        non_hr_hits = hits[i] - hrs[i]
        hr_tb = hrs[i] * 4
        non_hr_tb = 0
        for _ in range(non_hr_hits):
            r = rng.random()
            if r < 0.70:
                non_hr_tb += 1
            elif r < 0.90:
                non_hr_tb += 2
            else:
                non_hr_tb += 3
        tb[i] = hr_tb + non_hr_tb

    return {
        "mean_hits": round(hits.mean(), 2),
        "p_at_least_1_hit": round((hits >= 1).mean(), 3),
        "p_multi_hit": round((hits >= 2).mean(), 3),
        "mean_hr": round(hrs.mean(), 3),
        "p_at_least_1_hr": round((hrs >= 1).mean(), 3),
        "mean_tb": round(tb.mean(), 2),
        "p_3plus_tb": round((tb >= 3).mean(), 3),
    }


def run_simulations(player_probs: pd.DataFrame) -> pd.DataFrame:
    """Run game simulations for all players."""
    results = []
    for _, row in player_probs.iterrows():
        p_hit = row.get("p_hit_vs_lhp", row["p_hit_overall"])

        hr_rate_if_hit = 0.10
        sim = simulate_game(p_hit, hr_rate_if_hit)
        sim["name"] = row["name"]
        sim["p_hit_model"] = round(p_hit, 3)
        sim["n_pa_val"] = row["n_pa_val"]
        results.append(sim)

    return pd.DataFrame(results)


def plot_distributions(results: pd.DataFrame):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    results_sorted = results.sort_values("p_hit_model", ascending=True)
    names = results_sorted["name"].values

    ax = axes[0, 0]
    ax.barh(names, results_sorted["p_at_least_1_hit"])
    ax.set_xlabel("P(≥1 Hit)")
    ax.set_title("Probability of Getting at Least 1 Hit vs LHP")
    ax.set_xlim(0, 1)

    ax = axes[0, 1]
    ax.barh(names, results_sorted["p_multi_hit"])
    ax.set_xlabel("P(Multi-Hit Game)")
    ax.set_title("Probability of Multi-Hit Game vs LHP")
    ax.set_xlim(0, 1)

    ax = axes[1, 0]
    ax.barh(names, results_sorted["mean_tb"])
    ax.set_xlabel("Expected Total Bases")
    ax.set_title("Expected Total Bases per Game vs LHP")

    ax = axes[1, 1]
    ax.barh(names, results_sorted["p_at_least_1_hr"])
    ax.set_xlabel("P(≥1 HR)")
    ax.set_title("Probability of Hitting a HR vs LHP")
    ax.set_xlim(0, 0.5)

    plt.suptitle("LAD Hitters — Game Simulation vs LHP (10,000 sims per player)", fontsize=14)
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "game_sim_distributions.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Distribution chart saved → game_sim_distributions.png")


if __name__ == "__main__":
    print("=" * 60)
    print("  GAME SIMULATION (vs LHP)")
    print("=" * 60)

    model = load_model()
    val_df = pd.read_parquet(VAL_PATH)

    print("\nComputing player-level probabilities from 2025 val set...")
    player_probs = get_player_probs(model, val_df)

    print("\nRunning 10,000 game simulations per player...")
    results = run_simulations(player_probs)

    results = results.sort_values("p_at_least_1_hit", ascending=False)
    results.to_csv(REPORTS_DIR / "game_sim_results.csv", index=False)

    print(f"\nGame Simulation Results (vs LHP, {TYPICAL_PA_PER_GAME} PA/game):")
    print(results[["name", "p_hit_model", "p_at_least_1_hit", "p_multi_hit",
                    "mean_tb", "p_at_least_1_hr"]].to_string(index=False))

    plot_distributions(results)
    print(f"\nSaved → game_sim_results.csv")
