#!/usr/bin/env python3
"""
M1: Confidence meta-model as a probability stacker.

For each target (HR, Hit, XBH), train an HGBR Classifier on the 2025 val parquet
to predict the actual outcome from the base model's predicted probability PLUS
context features (BvP, rolling rates, pitcher data, month). The meta-model
output is a refined probability that improves over the base model when the
context features add information.

At inference time, picks are bucketed by **the meta-model probability** (not
residuals — earlier draft of this file did residual prediction, which just
re-sorts picks by raw P; that approach is wrong). Higher meta-prob = more
confident the outcome will fire.

  Lock   — top 10% by meta-prob (most confident pick will hit)
  Strong — next 15%
  Lean   — next 25%
  Avoid  — bottom 50%

Acceptance: realized rate is monotonic across buckets (Avoid < Lean < Strong < Lock).
Lock realized rate >= 1.5x baseline.

Persists model + thresholds to data/master/models/conf_meta/conf_meta_<target>.pkl
plus data/priors/conf_meta_thresholds.json.

Usage:
  python3 src/train_confidence_metamodel.py
  python3 src/train_confidence_metamodel.py --base-model beast
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import brier_score_loss, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from config import MASTER_DIR, REPORTS_DIR  # noqa: E402

VAL_PATH = MASTER_DIR / "features_val_league.parquet"
META_DIR = MASTER_DIR / "models" / "conf_meta"
META_DIR.mkdir(parents=True, exist_ok=True)
THRESHOLDS_JSON = ROOT / "data" / "priors" / "conf_meta_thresholds.json"
OUT_MD = REPORTS_DIR / "conf_meta_model_summary.md"

MODEL_PRED_PATHS = {
    "prod": REPORTS_DIR / "val_predictions_multi.parquet",
    "exp": REPORTS_DIR / "val_predictions_multi_exp.parquet",
    "beast": REPORTS_DIR / "val_predictions_multi_beast.parquet",
    "recency": REPORTS_DIR / "val_predictions_multi_recency.parquet",
}

# Inputs to the meta-model (must exist in val parquet OR be derivable)
META_FEATURE_COLS = [
    # the model's own prediction (added at runtime as p_target)
    "p_target",
    # BvP context
    "bvp_pa_count", "bvp_hr_count", "bvp_ba", "bvp_hr_rate", "bvp_has_history",
    # rolling rate context
    "roll_hr_rate_30", "roll_ba_30", "roll_k_rate_30", "roll_ev_30",
    "roll_hr_rate_100", "roll_ba_100",
    # pitcher data quality: number of pitches a pitcher has thrown (proxy)
    "p_velo_overall",
    # season time-of-year
    "month",
    # T2.5: cross-target consistency. The discrepancy between independent HR/XBH/Hit
    # predictions encodes confidence: when P(HR) is high but P(XBH) is low, the
    # prediction is internally confused and should be down-weighted.
    "hr_xbh_consistency",
    "xbh_hit_consistency",
]

# Target -> outcome col + display name
TARGET_SPECS = [
    ("hr", "is_hr", "HR"),
    ("hit", "is_hit", "Hit"),
    ("xbh", "is_xbh", "XBH"),
]

LABELS_DESC = ["Lock", "Strong", "Lean", "Avoid"]
# Higher meta-prob = more confident the outcome fires.
# Lock = top 10% by meta-prob, Strong = next 15%, Lean = next 25%, Avoid = remaining 50%.
# We split by quantile of meta-prob on the TRAINING set so bucket sizes are stable across slates.
QUANTILE_CUTS = [0.50, 0.75, 0.90]  # cutpoints in cumulative quantile (Avoid|Lean|Strong|Lock)


def _load_val(base_model: str) -> pd.DataFrame:
    val = pd.read_parquet(VAL_PATH)
    pred_path = MODEL_PRED_PATHS[base_model]
    preds = pd.read_parquet(pred_path)
    if len(preds) != len(val):
        raise RuntimeError(f"prediction file row count {len(preds)} != val {len(val)}")
    val = val.copy()
    for tgt in ("is_hr", "is_hit", "is_xbh"):
        val[f"p_{tgt}_basemodel"] = preds[f"p_{tgt}"].values
    val["month"] = pd.to_datetime(val["game_date"], errors="coerce").dt.month.fillna(0).astype(int)
    # T2.5: cross-target consistency ratios (sub-target / super-target). HR ⊂ XBH ⊂ Hit.
    # In a well-calibrated world: hr_xbh_consistency in (0, 1]; xbh_hit_consistency in (0, 1].
    # Values close to 1 mean the model is "leaning power"; values << 1 mean "leaning singles".
    eps = 1e-4
    val["hr_xbh_consistency"] = val["p_is_hr_basemodel"] / (val["p_is_xbh_basemodel"] + eps)
    val["xbh_hit_consistency"] = val["p_is_xbh_basemodel"] / (val["p_is_hit_basemodel"] + eps)
    return val


def _bucket_label(meta_prob: float, thresholds: dict) -> str:
    if meta_prob >= thresholds["lock"]:
        return "Lock"
    if meta_prob >= thresholds["strong"]:
        return "Strong"
    if meta_prob >= thresholds["lean"]:
        return "Lean"
    return "Avoid"


def train_one_target(val: pd.DataFrame, target: str, outcome_col: str, base_model: str) -> dict:
    val = val.copy()
    val["p_target"] = val[f"p_{outcome_col}_basemodel"]
    feat_cols = [c for c in META_FEATURE_COLS if c in val.columns]

    # 80/20 train/holdout (random; val parquet is already point-in-time-correct from H2 perspective)
    rng = np.random.default_rng(42)
    idx = rng.permutation(len(val))
    cut = int(0.8 * len(val))
    tr_idx, te_idx = idx[:cut], idx[cut:]
    X_tr = val.iloc[tr_idx][feat_cols].fillna(0)
    y_tr = val.iloc[tr_idx][outcome_col].astype(int)
    X_te = val.iloc[te_idx][feat_cols].fillna(0)
    y_te = val.iloc[te_idx][outcome_col].astype(int).values
    p_te = val.iloc[te_idx]["p_target"].values

    model = HistGradientBoostingClassifier(
        max_iter=200, max_depth=5, learning_rate=0.06,
        min_samples_leaf=80, random_state=42,
    )
    model.fit(X_tr, y_tr)

    meta_prob_te = model.predict_proba(X_te)[:, 1]

    # Build thresholds from TRAIN-set meta-probs so bucket sizes are predictable across slates
    meta_prob_tr = model.predict_proba(X_tr)[:, 1]
    # quantile cuts: split into [Avoid | Lean | Strong | Lock] at q=[0.50, 0.75, 0.90]
    q = np.quantile(meta_prob_tr, QUANTILE_CUTS)
    thresholds = {"lean": float(q[0]), "strong": float(q[1]), "lock": float(q[2])}

    bucket_te = np.array([_bucket_label(p, thresholds) for p in meta_prob_te])

    bucket_summary = []
    baseline = float(y_te.mean())
    for label in LABELS_DESC:
        mask = bucket_te == label
        n = int(mask.sum())
        if n == 0:
            bucket_summary.append({"label": label, "n": 0, "realized": float("nan"),
                                    "mean_meta_prob": float("nan"),
                                    "mean_base_prob": float("nan"),
                                    "lift": float("nan")})
            continue
        rate = float(y_te[mask].mean())
        bucket_summary.append({
            "label": label, "n": n, "realized": rate,
            "mean_meta_prob": float(meta_prob_te[mask].mean()),
            "mean_base_prob": float(p_te[mask].mean()),
            "lift": rate / baseline if baseline > 0 else float("nan"),
        })

    # Monotonicity check (Lock > Strong > Lean > Avoid by realized rate)
    ordered = [b["realized"] for b in bucket_summary
               if b["n"] > 0 and not np.isnan(b["realized"])]
    # bucket_summary is in order Lock, Strong, Lean, Avoid; for "monotonic high->low" we want
    # Lock >= Strong >= Lean >= Avoid
    monotonic = all(ordered[i] >= ordered[i + 1] for i in range(len(ordered) - 1)) if len(ordered) >= 2 else True

    # Compare meta-model Brier to base model Brier on the test set (does meta improve?)
    base_brier = brier_score_loss(y_te, p_te)
    meta_brier = brier_score_loss(y_te, meta_prob_te)
    base_auc = roc_auc_score(y_te, p_te) if y_te.sum() > 0 else float("nan")
    meta_auc = roc_auc_score(y_te, meta_prob_te) if y_te.sum() > 0 else float("nan")

    pkl_path = META_DIR / f"conf_meta_{target}.pkl"
    with open(pkl_path, "wb") as f:
        pickle.dump({"model": model, "feature_cols": feat_cols,
                      "thresholds": thresholds, "base_model": base_model}, f)

    return {
        "target": target,
        "feature_cols": feat_cols,
        "thresholds": thresholds,
        "n_train": int(len(tr_idx)),
        "n_test": int(len(te_idx)),
        "baseline": baseline,
        "buckets": bucket_summary,
        "monotonic": bool(monotonic),
        "base_brier": float(base_brier),
        "meta_brier": float(meta_brier),
        "brier_improvement": float(base_brier - meta_brier),
        "base_auc": float(base_auc),
        "meta_auc": float(meta_auc),
        "pkl_path": str(pkl_path),
    }


def render_md(results: list[dict]) -> str:
    lines = ["# M1: Confidence Meta-Model Summary (probability stacker)", ""]
    lines.append("Stacked HGBR Classifier on the 2025 val parquet with `beast` base predictions "
                 "+ context features (BvP, rolling rates, month, pitcher data). The meta-model "
                 "outputs a refined probability; Lock = top 10% by meta-prob.")
    lines.append("")

    for r in results:
        lines.append(f"## {r['target'].upper()}")
        lines.append(f"- Train rows: {r['n_train']:,}  |  Test rows: {r['n_test']:,}")
        lines.append(f"- Baseline realized rate (test): {r['baseline']:.4f}")
        thr = r['thresholds']
        lines.append(f"- Thresholds (meta-prob): "
                     f"Lock ≥ {thr['lock']:.4f}, Strong ≥ {thr['strong']:.4f}, "
                     f"Lean ≥ {thr['lean']:.4f}, else Avoid")
        lines.append(f"- Base model Brier: {r['base_brier']:.5f} → "
                     f"Meta Brier: {r['meta_brier']:.5f}  (Δ {r['brier_improvement']:+.6f})")
        lines.append(f"- Base AUC: {r['base_auc']:.4f}  →  Meta AUC: {r['meta_auc']:.4f}")
        lines.append(f"- Monotonic realized rate (Lock > Strong > Lean > Avoid): "
                     f"{'PASS' if r['monotonic'] else '**FAIL**'}")
        lines.append("")
        lines.append("| Bucket | n | Realized | Mean Meta-Prob | Mean Base-Prob | Lift |")
        lines.append("|---|---:|---:|---:|---:|---:|")
        for b in r["buckets"]:
            n = b["n"]
            if n == 0:
                lines.append(f"| {b['label']} | 0 | — | — | — | — |")
                continue
            lines.append(f"| {b['label']} | {n} | {b['realized']:.4f} | "
                         f"{b['mean_meta_prob']:.4f} | {b['mean_base_prob']:.4f} | "
                         f"{b['lift']:.2f}x |")
        lines.append("")

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train confidence meta-model.")
    parser.add_argument("--base-model", choices=list(MODEL_PRED_PATHS.keys()),
                        default="beast",
                        help="Which base model's predictions feed the meta-model (default: beast — H2 winner).")
    args = parser.parse_args()

    val = _load_val(args.base_model)
    results = []
    for target, outcome_col, _ in TARGET_SPECS:
        print(f"Training meta-model for {target} ...")
        r = train_one_target(val, target, outcome_col, args.base_model)
        results.append(r)

    THRESHOLDS_JSON.parent.mkdir(parents=True, exist_ok=True)
    THRESHOLDS_JSON.write_text(json.dumps({
        "_meta": {"base_model": args.base_model, "labels_desc": LABELS_DESC,
                   "quantile_cuts": QUANTILE_CUTS},
        "targets": {r["target"]: r["thresholds"] for r in results},
    }, indent=2))

    md = render_md(results)
    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUT_MD.write_text(md)
    print(md)
    print(f"\nWrote {OUT_MD}")
    print(f"Wrote {THRESHOLDS_JSON}")


if __name__ == "__main__":
    main()
