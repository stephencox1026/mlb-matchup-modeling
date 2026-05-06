#!/usr/bin/env python3
"""
H5 part 2: simple stacker that combines per-PA predictions from multiple base
models (beast, exp, prod, recency, lr_interactions) into a single calibrated
output via a held-out logistic regression on the val parquet.

The stacker only ships if its Brier improves on the best individual model with
a CI excluding 0 (acceptance gate from the audit memo).

Usage:
  python3 src/train_stacker.py
  python3 src/train_stacker.py --members beast exp lr_interactions
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from config import MASTER_DIR, REPORTS_DIR  # noqa: E402

VAL_PATH = MASTER_DIR / "features_val_league.parquet"
STACKER_DIR = MASTER_DIR / "models" / "stacker"
STACKER_DIR.mkdir(parents=True, exist_ok=True)
OUT_MD = REPORTS_DIR / "stacker_summary.md"

PRED_PATHS = {
    "prod": REPORTS_DIR / "val_predictions_multi.parquet",
    "exp": REPORTS_DIR / "val_predictions_multi_exp.parquet",
    "beast": REPORTS_DIR / "val_predictions_multi_beast.parquet",
    "recency": REPORTS_DIR / "val_predictions_multi_recency.parquet",
    "recent365": REPORTS_DIR / "val_predictions_multi_recent365.parquet",
    "lr_interactions": REPORTS_DIR / "val_predictions_multi_lr_interactions.parquet",
}

TARGETS = [("hr", "is_hr", "p_is_hr"),
           ("hit", "is_hit", "p_is_hit"),
           ("xbh", "is_xbh", "p_is_xbh")]


def _load(members: list[str]) -> dict:
    val = pd.read_parquet(VAL_PATH, columns=["batter", "pitcher", "game_pk", "game_date",
                                              "is_hr", "is_hit", "is_xbh"])
    preds = {}
    for m in members:
        if m not in PRED_PATHS:
            print(f"WARN: unknown member {m}", file=sys.stderr)
            continue
        if not PRED_PATHS[m].exists():
            print(f"WARN: missing {PRED_PATHS[m]}", file=sys.stderr)
            continue
        preds[m] = pd.read_parquet(PRED_PATHS[m]).reset_index(drop=True)
    return {"val": val.reset_index(drop=True), "preds": preds}


def _bootstrap_brier_diff(y, p_a, p_b, n_boot=200, seed=11):
    rng = np.random.default_rng(seed)
    n = len(y)
    base = (p_a - y) ** 2 - (p_b - y) ** 2
    diffs = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        diffs[i] = base[idx].mean()
    return float(base.mean()), float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))


def fit_target(val: pd.DataFrame, preds: dict, target: str, outcome_col: str, p_col: str) -> dict:
    members = list(preds.keys())
    X = np.column_stack([preds[m][p_col].values for m in members])
    y = val[outcome_col].astype(int).values

    rng = np.random.default_rng(42)
    idx = rng.permutation(len(y))
    cut = int(0.7 * len(y))
    tr_idx, te_idx = idx[:cut], idx[cut:]

    # Stacker = simple LR on the K member probabilities. With K=4-5 inputs this
    # has K weights + intercept; effectively a learned linear combination.
    stacker = LogisticRegression(C=1.0, max_iter=200)
    stacker.fit(X[tr_idx], y[tr_idx])
    stack_p = stacker.predict_proba(X[te_idx])[:, 1]

    # Compare stacker to best individual member on test set
    member_metrics = {}
    member_best = {"name": "(none)", "brier": float("inf")}
    for i, m in enumerate(members):
        pi = preds[m][p_col].values[te_idx]
        bri = brier_score_loss(y[te_idx], np.clip(pi, 1e-6, 1 - 1e-6))
        member_metrics[m] = {
            "brier": float(bri),
            "auc": float(roc_auc_score(y[te_idx], pi)) if y[te_idx].sum() > 0 else float("nan"),
        }
        if bri < member_best["brier"]:
            member_best = {"name": m, "brier": float(bri)}

    stack_brier = brier_score_loss(y[te_idx], np.clip(stack_p, 1e-6, 1 - 1e-6))
    stack_auc = roc_auc_score(y[te_idx], stack_p) if y[te_idx].sum() > 0 else float("nan")

    # Bootstrap CI on (stacker - best_individual) Brier diff
    p_best = preds[member_best["name"]][p_col].values[te_idx]
    mean_diff, lo, hi = _bootstrap_brier_diff(y[te_idx], stack_p, p_best)
    ships = lo < 0 and hi < 0  # both CI bounds < 0 → stacker beats best member with significance

    return {
        "target": target,
        "members": members,
        "weights": stacker.coef_.flatten().tolist(),
        "intercept": float(stacker.intercept_[0]),
        "stack_brier": float(stack_brier),
        "stack_auc": float(stack_auc),
        "best_member": member_best,
        "member_metrics": member_metrics,
        "brier_diff_vs_best": mean_diff,
        "brier_diff_ci": [lo, hi],
        "ships": bool(ships),
    }


def render_md(results: list[dict]) -> str:
    lines = ["# H5: Stacker Ensemble Summary", ""]
    lines.append("Logistic regression stacker over per-PA member predictions. "
                 "Acceptance gate: ship only if the stacker's Brier vs best individual member "
                 "has a 95% bootstrap CI **excluding 0** (i.e., stat-significantly lower).")
    lines.append("")
    lines.append("| Target | Members | Stack Brier | Best Member | Best Brier | "
                 "Δ Brier (Stack - Best) | 95% CI | Ships? |")
    lines.append("|---|---|---:|---|---:|---:|---|---|")
    for r in results:
        ci = r["brier_diff_ci"]
        ship = "**YES**" if r["ships"] else "no (CI includes 0)"
        lines.append(
            f"| {r['target']} | {', '.join(r['members'])} | {r['stack_brier']:.5f} | "
            f"{r['best_member']['name']} | {r['best_member']['brier']:.5f} | "
            f"{r['brier_diff_vs_best']:+.6f} | "
            f"[{ci[0]:+.6f}, {ci[1]:+.6f}] | {ship} |"
        )
    lines.append("")
    lines.append("## Stacker weights per target")
    lines.append("")
    for r in results:
        weight_pairs = [f"{m}: {w:+.3f}" for m, w in zip(r["members"], r["weights"])]
        lines.append(f"- **{r['target']}**: {', '.join(weight_pairs)}, "
                     f"intercept {r['intercept']:+.3f}")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train stacker ensemble for H5.")
    parser.add_argument("--members", nargs="+",
                        default=["beast", "exp", "prod", "recency", "recent365", "lr_interactions"],
                        help="Which model variants to stack.")
    args = parser.parse_args()

    data = _load(args.members)
    if not data["preds"]:
        print("No member predictions loaded; abort.", file=sys.stderr)
        sys.exit(1)

    results = []
    for tgt, outcome_col, p_col in TARGETS:
        print(f"Training stacker for {tgt} ...")
        r = fit_target(data["val"], data["preds"], tgt, outcome_col, p_col)
        results.append(r)

    # Persist a single bundle
    bundle = {"results": [{k: v for k, v in r.items() if k != "members"} for r in results],
              "members": args.members}
    (STACKER_DIR / "stacker_summary.json").write_text(json.dumps(bundle, indent=2))

    md = render_md(results)
    OUT_MD.write_text(md)
    print(md)
    print(f"\nWrote {OUT_MD}")


if __name__ == "__main__":
    main()
