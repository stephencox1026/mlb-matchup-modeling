#!/usr/bin/env python3
"""
H2: 2025 retrospective backtest comparing prod / exp / beast / recency.

The val parquet (data/master/features_val_league.parquet) is the entire 2025
regular season (Mar-Sep 2025, 183K PAs) with point-in-time-correct rolling
features. Each model variant has a corresponding val_predictions_multi*.parquet
holding its scored probabilities.

This script:
  1. Joins all four prediction files with the val outcomes
  2. Per-model: Brier, log loss, AUC, decile reliability, top-decile lift
  3. Pairwise model deltas with bootstrap CIs (paired comparisons by row)
  4. Saves data/reports/backtest_2025_summary.md and a per-target wide CSV

Output:
  data/reports/backtest_2025_summary.md
  data/reports/backtest_2025_per_model.csv
  data/reports/backtest_2025_decile_reliability.csv
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as scistats
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from config import MASTER_DIR, REPORTS_DIR  # noqa: E402

VAL_PATH = MASTER_DIR / "features_val_league.parquet"

MODEL_PREDICTION_PATHS = {
    "prod": REPORTS_DIR / "val_predictions_multi.parquet",
    "exp": REPORTS_DIR / "val_predictions_multi_exp.parquet",
    "beast": REPORTS_DIR / "val_predictions_multi_beast.parquet",
    "recency": REPORTS_DIR / "val_predictions_multi_recency.parquet",
    "recent365": REPORTS_DIR / "val_predictions_multi_recent365.parquet",
}

OUT_MD = REPORTS_DIR / "backtest_2025_summary.md"
OUT_PER_MODEL = REPORTS_DIR / "backtest_2025_per_model.csv"
OUT_DECILE = REPORTS_DIR / "backtest_2025_decile_reliability.csv"

TARGETS = ["is_hr", "is_hit", "is_xbh", "is_strikeout", "is_walk"]
PRIMARY_TARGETS = ["is_hr", "is_hit", "is_xbh"]


def _load() -> dict[str, pd.DataFrame]:
    val = pd.read_parquet(
        VAL_PATH,
        columns=["batter", "pitcher", "game_pk", "game_date", "events"] + TARGETS,
    )
    out = {}
    for name, path in MODEL_PREDICTION_PATHS.items():
        if not path.exists():
            print(f"WARN: missing {path}; skipping {name}", file=sys.stderr)
            continue
        preds = pd.read_parquet(path)
        # Align on (batter, pitcher, game_pk) — assumes same row order in val
        if len(preds) != len(val):
            print(f"WARN: {name} row count {len(preds)} != val {len(val)}; skipping",
                  file=sys.stderr)
            continue
        out[name] = preds
    return {"val": val, "models": out}


def _per_model_metrics(y: np.ndarray, p: np.ndarray) -> dict:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    if np.unique(y).size < 2:
        return {}
    return {
        "brier": brier_score_loss(y, p),
        "log_loss": log_loss(y, p),
        "auc": roc_auc_score(y, p),
        "ap": average_precision_score(y, p),
        "mean_pred": float(p.mean()),
        "realized": float(y.mean()),
    }


def _decile_table(y: np.ndarray, p: np.ndarray) -> pd.DataFrame:
    df = pd.DataFrame({"p": p, "y": y})
    df["dec"] = pd.qcut(df["p"], 10, labels=False, duplicates="drop")
    return df.groupby("dec", observed=True).agg(
        n=("y", "size"),
        mean_pred=("p", "mean"),
        realized=("y", "mean"),
    ).reset_index()


def _topdecile_lift(y: np.ndarray, p: np.ndarray) -> tuple[float, float, int]:
    """Returns (top_decile_realized, baseline_realized, n_top)."""
    df = pd.DataFrame({"p": p, "y": y})
    df["dec"] = pd.qcut(df["p"], 10, labels=False, duplicates="drop")
    top = df[df["dec"] == df["dec"].max()]
    return float(top["y"].mean()), float(df["y"].mean()), int(len(top))


def _bootstrap_brier_diff(y: np.ndarray, p_a: np.ndarray, p_b: np.ndarray,
                          n_boot: int = 200, seed: int = 7) -> tuple[float, float, float]:
    """Bootstrap 95% CI on Brier(a) - Brier(b). Negative => model A wins."""
    rng = np.random.default_rng(seed)
    n = len(y)
    diffs = np.empty(n_boot, dtype=float)
    sq_a = (p_a - y) ** 2
    sq_b = (p_b - y) ** 2
    base = sq_a - sq_b  # row-level squared-error difference
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        diffs[i] = base[idx].mean()
    return float(base.mean()), float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))


def run_backtest(window_n_boot: int = 200) -> dict:
    data = _load()
    val = data["val"]
    models = data["models"]
    if not models:
        return {"error": "no model prediction files found"}

    # Per-model metrics + decile reliability
    per_model = []
    decile_rows = []

    for tgt in TARGETS:
        if tgt not in val.columns:
            continue
        y = val[tgt].astype(int).values
        for mname, preds in models.items():
            p_col = f"p_{tgt}"
            if p_col not in preds.columns:
                continue
            p = preds[p_col].astype(float).values
            m = _per_model_metrics(y, p)
            if not m:
                continue
            top_realized, base, n_top = _topdecile_lift(y, p)
            per_model.append({
                "target": tgt, "model": mname, "n": int(len(y)),
                **m,
                "topdec_realized": top_realized,
                "topdec_lift": top_realized / base if base > 0 else float("nan"),
                "topdec_n": n_top,
            })
            dec = _decile_table(y, p)
            for _, dr in dec.iterrows():
                decile_rows.append({
                    "target": tgt, "model": mname, "decile": int(dr["dec"]),
                    "n": int(dr["n"]),
                    "mean_pred": float(dr["mean_pred"]),
                    "realized": float(dr["realized"]),
                })

    per_model_df = pd.DataFrame(per_model)
    decile_df = pd.DataFrame(decile_rows)

    # Pairwise Brier diffs for primary targets (HR, Hit, XBH)
    pairwise = []
    model_names = list(models.keys())
    for tgt in PRIMARY_TARGETS:
        if tgt not in val.columns:
            continue
        y = val[tgt].astype(int).values
        for i in range(len(model_names)):
            for j in range(i + 1, len(model_names)):
                a, b = model_names[i], model_names[j]
                pa = models[a].get(f"p_{tgt}")
                pb = models[b].get(f"p_{tgt}")
                if pa is None or pb is None:
                    continue
                mean_diff, lo, hi = _bootstrap_brier_diff(
                    y, pa.astype(float).values, pb.astype(float).values, n_boot=window_n_boot
                )
                pairwise.append({
                    "target": tgt, "model_a": a, "model_b": b,
                    "brier_diff_mean": mean_diff,
                    "brier_diff_ci_lo": lo, "brier_diff_ci_hi": hi,
                    "winner": ("tie" if (lo <= 0 <= hi)
                               else (a if mean_diff < 0 else b)),
                })
    pairwise_df = pd.DataFrame(pairwise)

    # Per-target winner by Brier and by top-decile lift
    leaderboards = []
    for tgt in PRIMARY_TARGETS:
        sub = per_model_df[per_model_df["target"] == tgt]
        if sub.empty:
            continue
        winner_brier = sub.loc[sub["brier"].idxmin(), "model"]
        winner_topdec = sub.loc[sub["topdec_lift"].idxmax(), "model"]
        leaderboards.append({
            "target": tgt,
            "winner_brier": winner_brier,
            "best_brier": float(sub["brier"].min()),
            "winner_topdec_lift": winner_topdec,
            "best_topdec_lift": float(sub["topdec_lift"].max()),
        })
    leader_df = pd.DataFrame(leaderboards)

    return {
        "per_model": per_model_df,
        "decile": decile_df,
        "pairwise": pairwise_df,
        "leaderboard": leader_df,
        "n_val": int(len(val)),
        "models_scored": model_names,
    }


def render_md(result: dict) -> str:
    if "error" in result:
        return f"# 2025 Backtest\n\n_{result['error']}_\n"

    per_model = result["per_model"]
    decile = result["decile"]
    pairwise = result["pairwise"]
    leader = result["leaderboard"]

    lines = [
        "# 2025 Retrospective Backtest (H2)",
        "",
        f"**Val window:** entire 2025 regular season ({result['n_val']:,} PAs)",
        f"**Models scored:** {', '.join(result['models_scored'])}",
        "",
        "## Leaderboard (per-target winners)",
        "",
        "| Target | Winner by Brier | Best Brier | Winner by Top-Decile Lift | Best Top-Decile Lift |",
        "|---|---|---:|---|---:|",
    ]
    for _, r in leader.iterrows():
        lines.append(
            f"| {r['target']} | **{r['winner_brier']}** | {r['best_brier']:.5f} | "
            f"**{r['winner_topdec_lift']}** | {r['best_topdec_lift']:.2f}x |"
        )
    lines.append("")

    lines.append("## Per-model metrics (all targets)")
    lines.append("")
    lines.append("| Target | Model | n | Brier | Log Loss | AUC | AP | "
                 "Mean Pred | Realized | Top-Dec Realized | Top-Dec Lift |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for _, r in per_model.iterrows():
        lines.append(
            f"| {r['target']} | {r['model']} | {r['n']:,} | "
            f"{r['brier']:.5f} | {r['log_loss']:.4f} | {r['auc']:.4f} | {r['ap']:.4f} | "
            f"{r['mean_pred']:.4f} | {r['realized']:.4f} | "
            f"{r['topdec_realized']:.4f} | {r['topdec_lift']:.2f}x |"
        )
    lines.append("")

    lines.append("## Pairwise model Brier deltas (bootstrap 95% CI on row-level squared-error diff)")
    lines.append("")
    lines.append("Negative diff = model_a beats model_b. CI excluding 0 = statistically distinguishable.")
    lines.append("")
    lines.append("| Target | A | B | Brier(A) - Brier(B) | 95% CI | Winner |")
    lines.append("|---|---|---|---:|---|---|")
    for _, r in pairwise.iterrows():
        wnr = "tie" if r["winner"] == "tie" else f"**{r['winner']}**"
        lines.append(
            f"| {r['target']} | {r['model_a']} | {r['model_b']} | "
            f"{r['brier_diff_mean']:+.6f} | "
            f"[{r['brier_diff_ci_lo']:+.6f}, {r['brier_diff_ci_hi']:+.6f}] | {wnr} |"
        )
    lines.append("")

    # Decile reliability per primary target
    lines.append("## Decile reliability (HR target)")
    lines.append("")
    hr_decile = decile[decile["target"] == "is_hr"]
    if not hr_decile.empty:
        wide = hr_decile.pivot_table(index="decile", columns="model",
                                     values=["mean_pred", "realized"]).round(4)
        lines.append(wide.to_markdown())
        lines.append("")

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="2025 retrospective backtest (H2).")
    parser.add_argument("--n-boot", type=int, default=200,
                        help="Bootstrap iterations for pairwise CI.")
    args = parser.parse_args()

    print("Loading val parquet + 4 model prediction files ...")
    result = run_backtest(window_n_boot=args.n_boot)
    if "error" in result:
        print(result["error"], file=sys.stderr)
        sys.exit(1)

    md = render_md(result)
    OUT_MD.write_text(md)
    result["per_model"].to_csv(OUT_PER_MODEL, index=False)
    result["decile"].to_csv(OUT_DECILE, index=False)

    print(md[:3000])
    print()
    print(f"Wrote {OUT_MD}")
    print(f"Wrote {OUT_PER_MODEL}")
    print(f"Wrote {OUT_DECILE}")


if __name__ == "__main__":
    main()
