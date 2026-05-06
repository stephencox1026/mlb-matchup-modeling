"""
Model monitoring stub.

Tracks prediction drift, calibration drift, and data freshness.
Designed to be run periodically (e.g., weekly during the season)
to detect when the model needs recalibration or retraining.
"""
import pickle
import numpy as np
import pandas as pd
from config import MASTER_DIR, REPORTS_DIR, RAW_DIR
from datetime import datetime


MODEL_PATH = MASTER_DIR / "models" / "best_model.pkl"
MONITOR_LOG = REPORTS_DIR / "monitor_log.csv"


def check_prediction_drift(new_pa_df: pd.DataFrame, model) -> dict:
    """Compare model predictions on new data vs. training distribution."""
    feat_cols = sorted([c for c in new_pa_df.columns if c.startswith((
        "roll_", "cum_", "vs_lhp", "month", "day_of_week", "pitch_count", "in_zone"
    ))])

    if not feat_cols:
        return {"status": "skip", "reason": "no feature columns found"}

    X = new_pa_df[feat_cols]
    preds = model.predict_proba(X)[:, 1]

    train_mean_pred = 0.242  # from training output
    new_mean_pred = preds.mean()
    drift = abs(new_mean_pred - train_mean_pred)

    return {
        "metric": "prediction_drift",
        "train_mean_pred": round(train_mean_pred, 4),
        "new_mean_pred": round(new_mean_pred, 4),
        "absolute_drift": round(drift, 4),
        "alert": drift > 0.03,
    }


def check_calibration(new_pa_df: pd.DataFrame, model) -> dict:
    """Check if predicted probabilities match actual outcomes."""
    feat_cols = sorted([c for c in new_pa_df.columns if c.startswith((
        "roll_", "cum_", "vs_lhp", "month", "day_of_week", "pitch_count", "in_zone"
    ))])

    if "is_hit" not in new_pa_df.columns or not feat_cols:
        return {"status": "skip", "reason": "missing target or features"}

    X = new_pa_df[feat_cols]
    preds = model.predict_proba(X)[:, 1]
    actual_rate = new_pa_df["is_hit"].mean()
    pred_rate = preds.mean()
    cal_gap = abs(actual_rate - pred_rate)

    return {
        "metric": "calibration_check",
        "actual_hit_rate": round(actual_rate, 4),
        "predicted_hit_rate": round(pred_rate, 4),
        "calibration_gap": round(cal_gap, 4),
        "alert": cal_gap > 0.02,
    }


def check_data_freshness() -> dict:
    """Check when model and data were last updated."""
    model_mtime = datetime.fromtimestamp(MODEL_PATH.stat().st_mtime) if MODEL_PATH.exists() else None
    pa_path = RAW_DIR / "statcast_pa_level.parquet"
    data_mtime = datetime.fromtimestamp(pa_path.stat().st_mtime) if pa_path.exists() else None

    now = datetime.now()
    model_age_days = (now - model_mtime).days if model_mtime else None
    data_age_days = (now - data_mtime).days if data_mtime else None

    return {
        "metric": "data_freshness",
        "model_last_trained": str(model_mtime) if model_mtime else "N/A",
        "data_last_updated": str(data_mtime) if data_mtime else "N/A",
        "model_age_days": model_age_days,
        "data_age_days": data_age_days,
        "alert": (model_age_days or 0) > 30 or (data_age_days or 0) > 7,
    }


def run_monitor(new_data_path: str = None):
    """Run all monitoring checks."""
    print("=" * 60)
    print("  MODEL MONITORING")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("=" * 60)

    model = pickle.load(open(MODEL_PATH, "rb")) if MODEL_PATH.exists() else None
    if model is None:
        print("  ERROR: No model found")
        return

    freshness = check_data_freshness()
    print(f"\n  Data freshness:")
    print(f"    Model age: {freshness['model_age_days']} days")
    print(f"    Data age:  {freshness['data_age_days']} days")
    if freshness["alert"]:
        print("    ⚠ ALERT: Model or data may be stale")

    if new_data_path:
        new_df = pd.read_parquet(new_data_path)
        drift = check_prediction_drift(new_df, model)
        print(f"\n  Prediction drift:")
        print(f"    Train mean: {drift.get('train_mean_pred')}")
        print(f"    New mean:   {drift.get('new_mean_pred')}")
        if drift.get("alert"):
            print("    ⚠ ALERT: Significant prediction drift detected")

        cal = check_calibration(new_df, model)
        print(f"\n  Calibration:")
        print(f"    Actual rate:    {cal.get('actual_hit_rate')}")
        print(f"    Predicted rate: {cal.get('predicted_hit_rate')}")
        if cal.get("alert"):
            print("    ⚠ ALERT: Calibration degraded")

    log_entry = {
        "timestamp": datetime.now().isoformat(),
        **freshness,
    }
    log_df = pd.DataFrame([log_entry])
    if MONITOR_LOG.exists():
        existing = pd.read_csv(MONITOR_LOG)
        log_df = pd.concat([existing, log_df], ignore_index=True)
    log_df.to_csv(MONITOR_LOG, index=False)
    print(f"\n  Monitor log updated → {MONITOR_LOG}")


if __name__ == "__main__":
    import sys
    data_path = sys.argv[1] if len(sys.argv) > 1 else None
    run_monitor(data_path)
