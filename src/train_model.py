"""
Ensemble modeling pipeline for PA-level hit prediction.

Models:
  1. Logistic Regression (baseline + regularized)
  2. HistGradientBoosting (sklearn built-in, LightGBM-style)
  3. Random Forest
  4. Stacking ensemble (meta-learner over the three)

All trained on 2015–2024, validated on 2025.
Outputs: calibrated probabilities, metrics, SHAP values, ablation results.
"""
import json
import pickle
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    RandomForestClassifier,
    StackingClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    brier_score_loss,
    log_loss,
    roc_auc_score,
    classification_report,
    average_precision_score,
)
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
import shap
from config import MASTER_DIR, REPORTS_DIR, FIGURES_DIR

warnings.filterwarnings("ignore", category=UserWarning)

TRAIN_PATH = MASTER_DIR / "features_train.parquet"
VAL_PATH = MASTER_DIR / "features_val.parquet"
TRAIN_PATH_LEAGUE = MASTER_DIR / "features_train_league.parquet"
VAL_PATH_LEAGUE = MASTER_DIR / "features_val_league.parquet"
MODEL_DIR = MASTER_DIR / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

USE_LEAGUE = False


def get_feature_cols(df: pd.DataFrame) -> list[str]:
    return sorted([c for c in df.columns if c.startswith((
        "roll_", "cum_", "vs_lhp", "month", "day_of_week", "pitch_count", "in_zone"
    ))])


def prepare_data(target: str = "is_hit"):
    tp = TRAIN_PATH_LEAGUE if USE_LEAGUE else TRAIN_PATH
    vp = VAL_PATH_LEAGUE if USE_LEAGUE else VAL_PATH
    train = pd.read_parquet(tp)
    val = pd.read_parquet(vp)

    feat_cols = get_feature_cols(train)
    print(f"Features: {len(feat_cols)}")
    print(f"Target: {target}")

    X_train = train[feat_cols].copy()
    y_train = train[target].copy()
    X_val = val[feat_cols].copy()
    y_val = val[target].copy()

    valid_train = y_train.notna() & X_train.notna().any(axis=1)
    valid_val = y_val.notna() & X_val.notna().any(axis=1)

    return (X_train[valid_train], y_train[valid_train],
            X_val[valid_val], y_val[valid_val],
            feat_cols, train[valid_train], val[valid_val])


def build_models(large_dataset: bool = False):
    """Build the three base models and the stacking ensemble."""

    rf_trees = 100 if large_dataset else 200
    rf_leaf = 50 if large_dataset else 20
    stack_cv = 3 if large_dataset else 5

    lr = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("lr", LogisticRegression(C=1.0, max_iter=1000, solver="lbfgs")),
    ])

    hgb = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("hgb", HistGradientBoostingClassifier(
            max_iter=300,
            max_depth=6,
            learning_rate=0.05,
            min_samples_leaf=50,
            l2_regularization=1.0,
            random_state=42,
        )),
    ])

    rf = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("rf", RandomForestClassifier(
            n_estimators=rf_trees,
            max_depth=10,
            min_samples_leaf=rf_leaf,
            random_state=42,
            n_jobs=-1,
        )),
    ])

    stack = StackingClassifier(
        estimators=[("lr", lr), ("hgb", hgb), ("rf", rf)],
        final_estimator=LogisticRegression(max_iter=500),
        cv=stack_cv,
        passthrough=False,
    )

    return {"logistic": lr, "hist_gbm": hgb, "random_forest": rf, "stack": stack}


def evaluate(name: str, model, X_val, y_val) -> dict:
    """Compute validation metrics for a fitted model."""
    y_prob = model.predict_proba(X_val)[:, 1]
    y_pred = model.predict(X_val)

    metrics = {
        "model": name,
        "brier": round(brier_score_loss(y_val, y_prob), 5),
        "log_loss": round(log_loss(y_val, y_prob), 5),
        "roc_auc": round(roc_auc_score(y_val, y_prob), 4),
        "avg_precision": round(average_precision_score(y_val, y_prob), 4),
        "mean_pred": round(y_prob.mean(), 4),
        "actual_rate": round(y_val.mean(), 4),
        "calibration_gap": round(abs(y_prob.mean() - y_val.mean()), 4),
    }
    return metrics


def calibrate_best(models: dict, X_train, y_train, X_val, y_val) -> dict:
    """Isotonic calibration. Uses fewer folds for large datasets."""
    from sklearn.model_selection import train_test_split

    n_train = len(X_train)
    n_folds = 2 if n_train > 200_000 else 5

    if n_train > 200_000:
        sample_size = min(200_000, n_train)
        _, X_cal, _, y_cal = train_test_split(
            X_train, y_train, test_size=sample_size, random_state=42, stratify=y_train
        )
    else:
        X_cal, y_cal = X_train, y_train

    results = {}
    for name, model in models.items():
        cal = CalibratedClassifierCV(model, method="isotonic", cv=n_folds)
        cal.fit(X_cal, y_cal)
        metrics = evaluate(f"{name}_calibrated", cal, X_val, y_val)
        results[name] = {"model": cal, "metrics": metrics}
    return results


def compute_shap(model, X_val, feat_cols, model_name: str, raw_models: dict = None):
    """Compute SHAP values for the validation set using the raw (uncalibrated) model."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    X_sample = X_val.sample(min(500, len(X_val)), random_state=42)
    imputer = SimpleImputer(strategy="median")
    X_filled = pd.DataFrame(imputer.fit_transform(X_sample), columns=feat_cols)

    try:
        raw_model = raw_models.get(model_name, model) if raw_models else model

        if hasattr(raw_model, "named_steps"):
            final_est = list(raw_model.named_steps.values())[-1]
            X_transformed = X_sample.copy()
            for step_name, step in list(raw_model.named_steps.items())[:-1]:
                X_transformed = pd.DataFrame(
                    step.transform(X_transformed), columns=feat_cols
                )
        else:
            final_est = raw_model
            X_transformed = X_filled

        explainer = shap.TreeExplainer(final_est)
        shap_values = explainer(X_transformed)

        if len(shap_values.shape) == 3:
            shap_values = shap_values[:, :, 1]

        fig, ax = plt.subplots(figsize=(10, 8))
        shap.plots.beeswarm(shap_values, show=False)
        plt.tight_layout()
        fig.savefig(FIGURES_DIR / f"shap_beeswarm_{model_name}.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  SHAP beeswarm saved → shap_beeswarm_{model_name}.png")

        fig, ax = plt.subplots(figsize=(10, 6))
        shap.plots.bar(shap_values, show=False)
        plt.tight_layout()
        fig.savefig(FIGURES_DIR / f"shap_bar_{model_name}.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  SHAP bar saved → shap_bar_{model_name}.png")

        vals = shap_values.values
        if vals.ndim == 1:
            vals = vals.reshape(-1, 1)
        importance = pd.DataFrame({
            "feature": feat_cols,
            "mean_abs_shap": np.abs(vals).mean(axis=0),
        }).sort_values("mean_abs_shap", ascending=False)
        importance.to_csv(REPORTS_DIR / f"shap_importance_{model_name}.csv", index=False)

        return shap_values

    except Exception as e:
        print(f"  SHAP failed for {model_name}: {e}")
        import traceback
        traceback.print_exc()
        return None


def run_ablations(models: dict, X_train, y_train, X_val, y_val, feat_cols) -> pd.DataFrame:
    """Run feature group ablation: drop each group and measure impact."""
    feature_groups = {
        "rolling_10": [c for c in feat_cols if c.endswith("_10")],
        "rolling_30": [c for c in feat_cols if c.endswith("_30")],
        "rolling_100": [c for c in feat_cols if c.endswith("_100")],
        "career_splits": [c for c in feat_cols if c.startswith("cum_")],
        "statcast_ev_la": [c for c in feat_cols if "ev" in c or "la" in c],
        "barrel": [c for c in feat_cols if "barrel" in c],
        "context": [c for c in feat_cols if c in ("vs_lhp", "month", "day_of_week", "pitch_count", "in_zone")],
    }

    results = []
    best_model_name = "hist_gbm"
    base_model_cls = models[best_model_name]

    for group_name, group_cols in feature_groups.items():
        if not group_cols:
            continue
        remaining = [c for c in feat_cols if c not in group_cols]
        if not remaining:
            continue

        try:
            from sklearn.base import clone
            ablated = clone(base_model_cls)
            ablated.fit(X_train[remaining], y_train)
            metrics = evaluate(f"ablate_{group_name}", ablated, X_val[remaining], y_val)
            metrics["dropped_group"] = group_name
            metrics["n_dropped"] = len(group_cols)
            results.append(metrics)
        except Exception as e:
            print(f"  Ablation {group_name} failed: {e}")

    return pd.DataFrame(results)


def compute_vif(X_train, feat_cols) -> pd.DataFrame:
    """Variance Inflation Factor for multicollinearity check."""
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    imputer = SimpleImputer(strategy="median")
    X_filled = pd.DataFrame(imputer.fit_transform(X_train[feat_cols]), columns=feat_cols)

    X_filled = X_filled.dropna(axis=1, how="all")
    clean_cols = X_filled.columns.tolist()

    vif_data = []
    for i, col in enumerate(clean_cols):
        try:
            vif = variance_inflation_factor(X_filled.values, i)
            vif_data.append({"feature": col, "VIF": round(vif, 2)})
        except Exception:
            vif_data.append({"feature": col, "VIF": np.nan})

    return pd.DataFrame(vif_data).sort_values("VIF", ascending=False)


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    print("=" * 60)
    print("  ML TRAINING PIPELINE")
    print("  Train: 2015–2024 | Val: 2025")
    print("=" * 60)

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    X_train, y_train, X_val, y_val, feat_cols, train_df, val_df = prepare_data("is_hit")
    print(f"\nTrain: {len(X_train):,} PAs, hit rate={y_train.mean():.3f}")
    print(f"Val:   {len(X_val):,} PAs, hit rate={y_val.mean():.3f}")

    # --- Baseline: constant predictor ---
    baseline_brier = brier_score_loss(y_val, np.full(len(y_val), y_train.mean()))
    print(f"\nBaseline (predict mean): Brier={baseline_brier:.5f}")

    # --- Train models ---
    large = len(X_train) > 200_000
    models = build_models(large_dataset=large)
    all_metrics = [{"model": "baseline_mean", "brier": round(baseline_brier, 5)}]

    for name, model in models.items():
        print(f"\nTraining {name}...")
        model.fit(X_train, y_train)
        metrics = evaluate(name, model, X_val, y_val)
        all_metrics.append(metrics)
        print(f"  Brier={metrics['brier']:.5f}  AUC={metrics.get('roc_auc','N/A')}  "
              f"AvgPrec={metrics.get('avg_precision','N/A')}  CalGap={metrics.get('calibration_gap','N/A')}")

    # --- Calibration ---
    print("\n--- Calibrating models ---")
    cal_results = calibrate_best(models, X_train, y_train, X_val, y_val)
    for name, res in cal_results.items():
        all_metrics.append(res["metrics"])
        m = res["metrics"]
        print(f"  {name}_cal: Brier={m['brier']:.5f}  AUC={m.get('roc_auc','N/A')}  CalGap={m.get('calibration_gap','N/A')}")

    metrics_df = pd.DataFrame(all_metrics)
    metrics_df.to_csv(REPORTS_DIR / "model_metrics.csv", index=False)
    print(f"\nMetrics saved → model_metrics.csv")

    # --- Pick best calibrated model ---
    cal_metrics = metrics_df[metrics_df["model"].str.contains("calibrated")]
    if not cal_metrics.empty:
        best_name = cal_metrics.loc[cal_metrics["brier"].idxmin(), "model"]
        best_model_key = best_name.replace("_calibrated", "")
        best_model = cal_results[best_model_key]["model"]
        print(f"\nBest calibrated model: {best_name}")
    else:
        best_model_key = "hist_gbm"
        best_model = models[best_model_key]
        best_name = best_model_key

    # Save best model
    with open(MODEL_DIR / "best_model.pkl", "wb") as f:
        pickle.dump(best_model, f)
    print(f"Saved best model → best_model.pkl")

    # --- SHAP ---
    print("\n--- SHAP Analysis ---")
    shap_values = compute_shap(best_model, X_val, feat_cols, best_model_key, raw_models=models)

    # --- VIF ---
    print("\n--- VIF (multicollinearity) ---")
    vif_df = compute_vif(X_train, feat_cols)
    vif_df.to_csv(REPORTS_DIR / "vif_scores.csv", index=False)
    print(f"Top VIF scores:")
    print(vif_df.head(10).to_string(index=False))

    # --- Ablation ---
    print("\n--- Feature Group Ablations ---")
    ablation_df = run_ablations(models, X_train, y_train, X_val, y_val, feat_cols)
    if not ablation_df.empty:
        ablation_df.to_csv(REPORTS_DIR / "ablation_results.csv", index=False)
        print(ablation_df[["dropped_group", "n_dropped", "brier", "roc_auc"]].to_string(index=False))

    # --- Calibration plot ---
    from sklearn.calibration import calibration_curve
    y_prob_best = best_model.predict_proba(X_val)[:, 1]
    fraction_pos, mean_pred = calibration_curve(y_val, y_prob_best, n_bins=10)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot([0, 1], [0, 1], "k--", label="Perfect calibration")
    ax.plot(mean_pred, fraction_pos, "s-", label=best_name)
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Fraction of positives")
    ax.set_title("Calibration Curve (2025 Validation)")
    ax.legend()
    fig.savefig(FIGURES_DIR / "calibration_curve.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nCalibration curve saved → calibration_curve.png")

    # --- Predictions on val set with player names ---
    val_pred = val_df.copy()
    val_pred["p_hit"] = best_model.predict_proba(X_val)[:, 1]
    val_pred.to_parquet(MASTER_DIR / "val_predictions.parquet", index=False)

    player_summary = val_pred.groupby("player_name_clean").agg(
        n_pa=("p_hit", "count"),
        actual_hit_rate=("is_hit", "mean"),
        predicted_hit_rate=("p_hit", "mean"),
        mean_p_hit=("p_hit", "mean"),
    ).round(3).sort_values("actual_hit_rate", ascending=False)
    player_summary.to_csv(REPORTS_DIR / "player_val_predictions.csv")
    print(f"\nPlayer-level 2025 validation predictions:")
    print(player_summary.to_string())

    print(f"\n{'='*60}")
    print(f"  TRAINING COMPLETE")
    print(f"{'='*60}")
    print(f"Best model: {best_name}")
    print(f"Best Brier: {cal_results[best_model_key]['metrics']['brier']:.5f}")
    print(f"Best AUC:   {cal_results[best_model_key]['metrics']['roc_auc']:.4f}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--league", action="store_true", help="Use league-wide training data")
    args = parser.parse_args()
    if args.league:
        USE_LEAGUE = True
    main()
