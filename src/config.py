"""Project-wide constants and roster definition."""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
MASTER_DIR = DATA_DIR / "master"
# Experiment bundle (bpt barrel + roll_est_woba features); optional second models + val features.
EXPERIMENT_MODEL_SUBDIR = "exp_bpt_xwoba"
EXPERIMENT_MODEL_DIR = MASTER_DIR / "models" / EXPERIMENT_MODEL_SUBDIR
EXPERIMENT_VAL_FEATURES = MASTER_DIR / "features_val_league_exp.parquet"
# Third bundle: L3/L5 short rolls + optional recency-weighted training (see docs/recency-model.md).
# DEPRECATED: replaced by recent365 model (data-window diversity, same arch as beast). Kept here
# for back-compat; new code should reference RECENT365_* below.
RECENCY_MODEL_SUBDIR = "exp_recency_l3l5"
RECENCY_MODEL_DIR = MASTER_DIR / "models" / RECENCY_MODEL_SUBDIR
RECENCY_VAL_FEATURES = MASTER_DIR / "features_val_league_recency.parquet"

# Recent365 bundle: same XGBoost arch as beast trained on rolling 365-day window.
# Replaces recency in production for the dual-model dashboard tab. recent365 wins
# Hit decisively over recency (Brier -0.0005, AUC +0.0035) and is equivalent
# elsewhere; H5 stacker still fails its CI gate so the next ensemble swing is
# PyTorch MLP (algorithmic diversity, not data-window diversity).
RECENT365_MODEL_SUBDIR = "recent365"
RECENT365_MODEL_DIR = MASTER_DIR / "models" / RECENT365_MODEL_SUBDIR
# recent365 was trained on the beast-feature val parquet (park + xptw + month_sin/cos)
# so its inference val features are the beast val parquet, not a custom one.
RECENT365_VAL_FEATURES = MASTER_DIR / "features_val_league_beast.parquet"
# Beast bundle: HR-focused superset (optional third model stream for dashboard tab).
BEAST_MODEL_SUBDIR = "exp_beast"
BEAST_MODEL_DIR = MASTER_DIR / "models" / BEAST_MODEL_SUBDIR
BEAST_VAL_FEATURES = MASTER_DIR / "features_val_league_beast.parquet"
REPORTS_DIR = DATA_DIR / "reports"
TRACKING_DIR = DATA_DIR / "tracking"
FIGURES_DIR = PROJECT_ROOT / "docs" / "figures"
DOCS_DIR = PROJECT_ROOT / "docs"

# Pseudo-PA weight when shrinking BvP HR rate toward league (higher = less influence from
# 1 HR in a handful of PA). Keep in sync with `build_matchup_features.py` HR shrink.
BVP_HR_BAYES_PRIOR_WEIGHT = 250

TEAM_ABBR = "LAD"
SEASON = 2026

ROSTER_14 = [
    {"name": "Max Muncy",            "pos": "3B"},
    {"name": "Shohei Ohtani",        "pos": "TWP"},
    {"name": "Freddie Freeman",      "pos": "1B"},
    {"name": "Dalton Rushing",       "pos": "C"},
    {"name": "Kyle Tucker",          "pos": "RF"},
    {"name": "Mookie Betts",         "pos": "SS"},
    {"name": "Alex Call",            "pos": "RF"},
    {"name": "Santiago Espinal",     "pos": "3B"},
    {"name": "Alex Freeland",        "pos": "2B"},
    {"name": "Teoscar Hernández",    "pos": "RF"},
    {"name": "Hyeseong Kim",         "pos": "2B"},
    {"name": "Andy Pages",           "pos": "CF"},
    {"name": "Miguel Rojas",         "pos": "2B"},
    {"name": "Will Smith",           "pos": "C"},
]

PLAYER_NAMES = [p["name"] for p in ROSTER_14]
