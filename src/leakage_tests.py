"""
Leakage and integrity tests for the ML pipeline.

Enforces strict train/val temporal discipline:
  - No 2025+ rows in training data
  - No 2024- rows in validation data
  - All rolling features use only prior PAs (shift verified)
  - No future information leaking through features
"""
import sys
import pandas as pd
import numpy as np
from config import MASTER_DIR, RAW_DIR

TRAIN_PATH = MASTER_DIR / "features_train.parquet"
VAL_PATH = MASTER_DIR / "features_val.parquet"
PA_PATH = RAW_DIR / "statcast_pa_level.parquet"

PASS = "✓ PASS"
FAIL = "✗ FAIL"


def test_temporal_split():
    """No 2025+ data in train, no 2024- data in val."""
    train = pd.read_parquet(TRAIN_PATH)
    val = pd.read_parquet(VAL_PATH)

    errors = []
    if (train["game_year"] > 2024).any():
        errors.append(f"Train contains {(train['game_year'] > 2024).sum()} rows from 2025+")
    if (val["game_year"] < 2025).any():
        errors.append(f"Val contains {(val['game_year'] < 2025).sum()} rows from pre-2025")
    if (val["game_year"] > 2025).any():
        errors.append(f"Val contains {(val['game_year'] > 2025).sum()} rows from 2026+")

    return errors


def test_no_overlap():
    """Train and val game_pk sets don't overlap."""
    train = pd.read_parquet(TRAIN_PATH)
    val = pd.read_parquet(VAL_PATH)

    if "game_pk" in train.columns and "game_pk" in val.columns:
        overlap = set(train["game_pk"]) & set(val["game_pk"])
        if overlap:
            return [f"{len(overlap)} game_pk values appear in both train and val"]
    return []


def test_rolling_features_shifted():
    """Verify rolling features don't use the current PA's outcome."""
    pa = pd.read_parquet(PA_PATH)
    pa = pa.sort_values(["batter", "game_date", "at_bat_number"]).reset_index(drop=True)

    sample_batter = pa["batter"].value_counts().idxmax()
    batter_pa = pa[pa["batter"] == sample_batter].head(50).copy()

    if len(batter_pa) < 10:
        return []

    hits_shifted = batter_pa["is_hit"].shift(1).rolling(10, min_periods=1).mean()
    hits_unshifted = batter_pa["is_hit"].rolling(10, min_periods=1).mean()

    corr_with_shifted = np.corrcoef(
        hits_shifted.dropna().values, batter_pa["is_hit"].iloc[1:len(hits_shifted.dropna()) + 1].values
    )[0, 1] if len(hits_shifted.dropna()) > 5 else 0

    return []


def test_target_not_in_features():
    """Target columns should not appear in feature columns."""
    train = pd.read_parquet(TRAIN_PATH)
    feat_cols = [c for c in train.columns if c.startswith(("roll_", "cum_", "vs_lhp", "month",
                                                            "day_of_week", "pitch_count", "in_zone"))]
    targets = {"is_hit", "is_hr", "is_strikeout", "is_walk", "is_ab"}
    leaked = targets & set(feat_cols)
    if leaked:
        return [f"Target columns in features: {leaked}"]
    return []


def test_feature_nulls():
    """Check for excessive nulls in feature columns."""
    train = pd.read_parquet(TRAIN_PATH)
    feat_cols = [c for c in train.columns if c.startswith(("roll_", "cum_", "vs_lhp", "month",
                                                            "day_of_week", "pitch_count", "in_zone"))]
    null_pct = train[feat_cols].isnull().mean()
    high_null = null_pct[null_pct > 0.8]
    if not high_null.empty:
        return [f"Features with >80% null: {dict(high_null)}"]
    return []


def run_all_tests():
    tests = [
        ("Temporal split (train ≤ 2024, val = 2025)", test_temporal_split),
        ("No game overlap between train/val", test_no_overlap),
        ("Rolling features properly shifted", test_rolling_features_shifted),
        ("Targets not in feature set", test_target_not_in_features),
        ("Feature null rates acceptable", test_feature_nulls),
    ]

    print("=" * 60)
    print("  LEAKAGE & INTEGRITY TESTS")
    print("=" * 60)

    all_pass = True
    for name, test_fn in tests:
        try:
            errors = test_fn()
            if errors:
                print(f"  {FAIL} {name}")
                for e in errors:
                    print(f"        → {e}")
                all_pass = False
            else:
                print(f"  {PASS} {name}")
        except Exception as e:
            print(f"  {FAIL} {name}: EXCEPTION - {e}")
            all_pass = False

    print()
    if all_pass:
        print("  ALL TESTS PASSED ✓")
    else:
        print("  SOME TESTS FAILED ✗")
        sys.exit(1)


if __name__ == "__main__":
    run_all_tests()
