"""
Runtime helpers for the M1 confidence meta-model.

Loads the pickled per-target HGBR Classifier + bucket thresholds and exposes
`compute_meta_label(target, p_base, features_dict)` which returns
(label_name, meta_probability).

Used by src/narrative_engine.py at inference time to attach Lock/Strong/Lean/Avoid
labels alongside the legacy hand-tuned conf_*_label values.
"""
from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
META_DIR = ROOT / "data" / "master" / "models" / "conf_meta"
THRESHOLDS_JSON = ROOT / "data" / "priors" / "conf_meta_thresholds.json"

LABELS_DESC = ["Lock", "Strong", "Lean", "Avoid"]


_MODEL_CACHE: dict[str, Any] = {}
_THRESHOLDS_CACHE: dict | None = None
_THRESHOLDS_MTIME: float | None = None


def _load_thresholds() -> dict | None:
    global _THRESHOLDS_CACHE, _THRESHOLDS_MTIME
    if not THRESHOLDS_JSON.exists():
        return None
    mtime = THRESHOLDS_JSON.stat().st_mtime
    if _THRESHOLDS_CACHE is not None and _THRESHOLDS_MTIME == mtime:
        return _THRESHOLDS_CACHE
    _THRESHOLDS_CACHE = json.loads(THRESHOLDS_JSON.read_text())
    _THRESHOLDS_MTIME = mtime
    return _THRESHOLDS_CACHE


def _load_model(target: str) -> dict | None:
    if target in _MODEL_CACHE:
        return _MODEL_CACHE[target]
    pkl = META_DIR / f"conf_meta_{target}.pkl"
    if not pkl.exists():
        _MODEL_CACHE[target] = None
        return None
    with open(pkl, "rb") as f:
        bundle = pickle.load(f)
    _MODEL_CACHE[target] = bundle
    return bundle


def _bucket(meta_prob: float, thresholds: dict) -> str:
    if meta_prob >= thresholds["lock"]:
        return "Lock"
    if meta_prob >= thresholds["strong"]:
        return "Strong"
    if meta_prob >= thresholds["lean"]:
        return "Lean"
    return "Avoid"


def compute_meta_label(target: str, p_base: float,
                       features_dict: dict[str, float | int | None],
                       *, p_hr: float | None = None,
                       p_xbh: float | None = None,
                       p_hit: float | None = None) -> tuple[str, float]:
    """Score the M1 meta-model and return (label, meta_probability).

    Args:
      target: 'hr', 'hit', or 'xbh'
      p_base: the base model's predicted probability for this target
      features_dict: dict of feature values; missing features default to 0
      p_hr / p_xbh / p_hit: optional cross-target predictions used to compute
        the T2.5 hr_xbh_consistency and xbh_hit_consistency features. When
        unspecified, these features default to 1.0 (neutral signal).

    Returns:
      (label, meta_prob). On any failure (model not trained, bad input),
      returns ("Avoid", float(p_base)) so callers see a degraded but
      sensible result rather than a crash.
    """
    bundle = _load_model(target)
    thresh_all = _load_thresholds()
    if bundle is None or thresh_all is None:
        return ("Avoid", float(p_base))
    thresholds = thresh_all.get("targets", {}).get(target)
    if thresholds is None:
        return ("Avoid", float(p_base))

    feat_cols = bundle["feature_cols"]
    model = bundle["model"]

    feats = features_dict.copy() if features_dict else {}
    feats["p_target"] = float(p_base)
    # T2.5: consistency features (defaults are neutral if cross-target unavailable)
    eps = 1e-4
    if p_hr is not None and p_xbh is not None:
        feats["hr_xbh_consistency"] = float(p_hr) / (float(p_xbh) + eps)
    elif "hr_xbh_consistency" not in feats:
        feats["hr_xbh_consistency"] = 1.0
    if p_xbh is not None and p_hit is not None:
        feats["xbh_hit_consistency"] = float(p_xbh) / (float(p_hit) + eps)
    elif "xbh_hit_consistency" not in feats:
        feats["xbh_hit_consistency"] = 1.0

    row = {}
    for c in feat_cols:
        v = feats.get(c, 0.0)
        if v is None or (isinstance(v, float) and np.isnan(v)):
            v = 0.0
        row[c] = float(v)
    X = pd.DataFrame([row], columns=feat_cols)

    try:
        meta_p = float(model.predict_proba(X)[:, 1][0])
    except Exception:
        return ("Avoid", float(p_base))

    return (_bucket(meta_p, thresholds), meta_p)


def reset_cache() -> None:
    """Force re-load of pickled models + thresholds on next call."""
    global _THRESHOLDS_CACHE, _THRESHOLDS_MTIME
    _MODEL_CACHE.clear()
    _THRESHOLDS_CACHE = None
    _THRESHOLDS_MTIME = None
