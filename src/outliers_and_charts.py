"""Compute outlier scores (1–5) for 2026 vs career same-split, generate charts."""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from config import MASTER_DIR, REPORTS_DIR, FIGURES_DIR

FIGURES_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "#f8f9fa",
    "axes.grid": True,
    "grid.alpha": 0.3,
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
})


def load_master() -> pd.DataFrame:
    return pd.read_parquet(MASTER_DIR / "lad_hitters_sprint.parquet")


def compute_outlier_scores(master: pd.DataFrame) -> pd.DataFrame:
    """Score each player's 2026 split vs their own career same-split on a 1-5 scale.

    The score reflects how far 2026 OPS deviates from career OPS in the same
    split. 1 = close to career norm, 5 = extreme outlier. PA < 10 gets a
    low-confidence flag.
    """
    rows = []
    for split in ["vs LHP", "vs RHP"]:
        s2026 = master[(master["scope"] == "2026") & (master["split"] == split)]
        career = master[(master["scope"] == "Career") & (master["split"] == split)]

        merged = s2026.merge(career, on=["name", "pos", "mlbam_id"],
                             suffixes=("_2026", "_career"))

        for _, r in merged.iterrows():
            ops_2026 = r["OPS_2026"]
            ops_career = r["OPS_career"]
            pa_2026 = r["PA_2026"]
            pa_career = r["PA_career"]

            if ops_career > 0:
                diff = ops_2026 - ops_career
                pct_diff = abs(diff / ops_career)
            else:
                diff = ops_2026
                pct_diff = 1.0 if ops_2026 > 0 else 0.0

            if pct_diff < 0.10:
                score = 1
            elif pct_diff < 0.25:
                score = 2
            elif pct_diff < 0.40:
                score = 3
            elif pct_diff < 0.60:
                score = 4
            else:
                score = 5

            rows.append({
                "name": r["name"],
                "split": split,
                "OPS_2026": round(ops_2026, 3),
                "OPS_career": round(ops_career, 3),
                "OPS_diff": round(diff, 3),
                "pct_diff": round(pct_diff * 100, 1),
                "outlier_score": score,
                "PA_2026": pa_2026,
                "PA_career": pa_career,
                "low_n_flag": "LOW-N" if pa_2026 < 10 else "",
            })

    return pd.DataFrame(rows).sort_values(
        ["split", "outlier_score"], ascending=[True, False]
    ).reset_index(drop=True)


def plot_ops_lhp_vs_rhp(master: pd.DataFrame):
    """Bar chart comparing 2026 OPS vs LHP and vs RHP for each batter."""
    s2026 = master[master["scope"] == "2026"].copy()
    lhp = s2026[s2026["split"] == "vs LHP"][["name", "OPS"]].rename(columns={"OPS": "vs LHP"})
    rhp = s2026[s2026["split"] == "vs RHP"][["name", "OPS"]].rename(columns={"OPS": "vs RHP"})
    df = lhp.merge(rhp, on="name").sort_values("vs LHP", ascending=True)

    fig, ax = plt.subplots(figsize=(10, 7))
    y = np.arange(len(df))
    ax.barh(y - 0.17, df["vs LHP"], 0.34, label="vs LHP", color="#2176AE", alpha=0.85)
    ax.barh(y + 0.17, df["vs RHP"], 0.34, label="vs RHP", color="#E8575A", alpha=0.85)
    ax.set_yticks(y)
    ax.set_yticklabels(df["name"])
    ax.set_xlabel("OPS (2026)")
    ax.set_title("LAD Hitters — 2026 OPS: vs LHP vs vs RHP")
    ax.legend(loc="lower right")
    ax.axvline(x=0.700, color="gray", ls="--", lw=0.8, alpha=0.5)
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "ops_lhp_vs_rhp_2026.png", dpi=150)
    plt.close(fig)
    print("  Saved ops_lhp_vs_rhp_2026.png")


def plot_outlier_scores(scores: pd.DataFrame):
    """Horizontal bar chart of outlier scores, faceted by split."""
    for split in ["vs LHP", "vs RHP"]:
        sub = scores[scores["split"] == split].sort_values("outlier_score")
        fig, ax = plt.subplots(figsize=(9, 6))
        colors = sub["outlier_score"].map({
            1: "#4CAF50", 2: "#8BC34A", 3: "#FFC107", 4: "#FF9800", 5: "#F44336"
        })
        bars = ax.barh(sub["name"], sub["outlier_score"], color=colors, alpha=0.85)
        for bar, row in zip(bars, sub.itertuples()):
            label = f"{row.outlier_score}"
            if row.low_n_flag:
                label += " *"
            ax.text(bar.get_width() + 0.08, bar.get_y() + bar.get_height()/2,
                    label, va="center", fontsize=10)
        ax.set_xlabel("Outlier Score (1=normal, 5=extreme)")
        tag = "LHP" if "LHP" in split else "RHP"
        ax.set_title(f"2026 vs Career Outlier Score — {split}\n(* = low PA)")
        ax.set_xlim(0, 5.8)
        plt.tight_layout()
        fig.savefig(FIGURES_DIR / f"outlier_score_{tag.lower()}.png", dpi=150)
        plt.close(fig)
        print(f"  Saved outlier_score_{tag.lower()}.png")


def plot_statcast_ev(summary_path=None):
    """Bar chart of avg exit velocity from Statcast contact summary."""
    if summary_path is None:
        summary_path = REPORTS_DIR / "statcast_contact_summary.csv"
    if not summary_path.exists():
        print("  Skipping EV chart — no Statcast summary")
        return
    df = pd.read_csv(summary_path).sort_values("avg_EV", ascending=True)

    fig, ax = plt.subplots(figsize=(9, 6))
    colors = ["#2176AE" if ev >= 90 else "#90CAF9" for ev in df["avg_EV"]]
    ax.barh(df["name"], df["avg_EV"], color=colors, alpha=0.85)
    for i, (name, ev, hh) in enumerate(zip(df["name"], df["avg_EV"], df["hard_hit_pct"])):
        ax.text(ev + 0.3, i, f"{ev:.1f} ({hh:.0f}% hard)", va="center", fontsize=9)
    ax.set_xlabel("Avg Exit Velocity (mph)")
    ax.set_title("LAD Hitters — 2026 Statcast Exit Velocity\n(hard-hit % in parens)")
    ax.axvline(x=90, color="gray", ls="--", lw=0.8, alpha=0.5)
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "statcast_ev_2026.png", dpi=150)
    plt.close(fig)
    print("  Saved statcast_ev_2026.png")


def plot_peterson_matchup(master: pd.DataFrame):
    """Bar chart of career PA and OPS vs David Peterson."""
    pet = master[(master["split"] == "vs Peterson") & (master["PA"] > 0)].copy()
    if pet.empty:
        print("  Skipping Peterson chart — no data with PA>0")
        return
    pet = pet.sort_values("OPS", ascending=True)

    fig, ax1 = plt.subplots(figsize=(9, 5))
    y = np.arange(len(pet))
    ax1.barh(y, pet["PA"], color="#78909C", alpha=0.7, label="Career PA")
    ax1.set_yticks(y)
    ax1.set_yticklabels(pet["name"])
    ax1.set_xlabel("Career PA vs Peterson")

    ax2 = ax1.twiny()
    ax2.plot(pet["OPS"], y, "D", color="#E8575A", markersize=8, label="OPS")
    ax2.set_xlabel("OPS", color="#E8575A")

    ax1.set_title("LAD Hitters — Career vs David Peterson")
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="lower right")
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "peterson_matchup.png", dpi=150)
    plt.close(fig)
    print("  Saved peterson_matchup.png")


if __name__ == "__main__":
    master = load_master()

    print("Computing outlier scores...")
    scores = compute_outlier_scores(master)
    scores.to_csv(REPORTS_DIR / "outlier_scores.csv", index=False)
    print(scores.to_string(index=False))

    print("\nGenerating charts...")
    plot_ops_lhp_vs_rhp(master)
    plot_outlier_scores(scores)
    plot_statcast_ev()
    plot_peterson_matchup(master)
    print("\nDone — all charts in docs/figures/")
