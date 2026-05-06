"""
Predict runs allowed by each probable starter using the trained starter-runs model.

Loads the HistGBR model trained on ~40K historical starter outings (actual runs),
builds game-level features (pitcher profile + aggregated lineup), and produces
point estimates with conformal prediction intervals.

Confidence = interval width from conformal calibration, not hand-tuned multipliers.
"""
from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import RAW_DIR, MASTER_DIR, REPORTS_DIR
from build_starter_run_training import (
    BATTER_AGG_FEATURES,
    PITCHER_PROFILE_COLS,
)

MODEL_DIR = MASTER_DIR / "models"

DEFAULT_COVERAGE = "50"

CONF_THRESHOLDS = {
    "High": 2.5,
    "Medium": 3.0,
    "Low": 3.5,
}


def _confidence_label(width: float) -> str:
    for label, threshold in CONF_THRESHOLDS.items():
        if width < threshold:
            return label
    return "Very Low"


def load_starter_runs_model():
    with open(MODEL_DIR / "starter_runs_model.pkl", "rb") as f:
        return pickle.load(f)


def load_conformal_quantiles() -> dict:
    with open(MODEL_DIR / "starter_runs_conformal.json") as f:
        return json.load(f)


def load_feature_cols() -> list[str]:
    with open(MODEL_DIR / "starter_runs_feature_cols.json") as f:
        return json.load(f)


def load_pitcher_profiles() -> pd.DataFrame:
    path = RAW_DIR / "pitcher_profiles_by_season.parquet"
    if path.exists():
        return pd.read_parquet(path)
    return pd.DataFrame()


def load_qualifying_pa_map() -> dict[int, int]:
    path = RAW_DIR / "qualifying_batters_2026.csv"
    df = pd.read_csv(path)
    return {int(r.mlbam_id): int(r.PA) for _, r in df.iterrows()}


def top9_lineup(roster_ids: list[int], pa_map: dict[int, int]) -> list[int]:
    scored = [(bid, pa_map.get(bid, 0)) for bid in roster_ids]
    scored.sort(key=lambda x: -x[1])
    return [b for b, _ in scored[:9]]


def _get_pitcher_features(pitcher_id: int, profiles: pd.DataFrame) -> dict:
    pp = profiles[profiles["pitcher"] == pitcher_id]
    if pp.empty:
        return {}
    latest = pp.sort_values("season").iloc[-1]
    return {f"sp_{c}": latest[c] for c in PITCHER_PROFILE_COLS if c in latest.index}


def _aggregate_lineup_from_matchups(
    lineup: list[int],
    pitcher_id: int,
    matchup_lookup: dict[tuple[int, int], dict],
) -> dict:
    """Build lineup aggregation features from the already-computed matchup predictions.

    Uses the per-batter matchup dicts which carry raw model probs and the
    narrative engine's feature extraction already performed. We extract the
    batter-level stats embedded in the matchup results.
    """
    batter_stats = []
    for bid in lineup:
        m = matchup_lookup.get((bid, pitcher_id))
        if m is None:
            continue
        row = {
            "roll_ba_30": m.get("career_hit", 0.222),
            "roll_ba_100": m.get("career_hit", 0.222),
            "roll_hr_rate_30": m.get("career_hr", 0.031),
            "roll_hr_rate_100": m.get("career_hr", 0.031),
            "roll_k_rate_30": m.get("career_k", 0.222),
            "roll_k_rate_100": m.get("career_k", 0.222),
            "bvp_ba": 0.243,
            "bvp_hr_rate": 0.031,
            "bvp_pa_count": 0.0,
            "bvp_has_history": 0,
        }
        batter_stats.append(row)

    if not batter_stats:
        return {"lineup_n_batters": 0}

    lf = pd.DataFrame(batter_stats)
    result = {}
    for col in lf.columns:
        vals = lf[col].dropna()
        if vals.empty:
            continue
        result[f"lineup_mean_{col}"] = float(vals.mean())
        result[f"lineup_max_{col}"] = float(vals.max())
        if len(vals) >= 3:
            result[f"lineup_std_{col}"] = float(vals.std())

    result["lineup_n_batters"] = len(batter_stats)
    result["lineup_n_with_bvp"] = int(lf["bvp_has_history"].sum())
    result["lineup_mean_bvp_pa"] = float(lf["bvp_pa_count"].mean())
    return result


def _aggregate_lineup_from_pa_features(
    lineup: list[int],
    pa_features: pd.DataFrame,
    game_date: pd.Timestamp,
) -> dict:
    """Build lineup features from the PA-level feature parquets (richer, used when available)."""
    lineup_feats = []
    for bid in lineup:
        grp = pa_features[pa_features["batter"] == bid]
        if grp.empty:
            continue
        prior = grp[grp["game_date"] < game_date]
        if prior.empty:
            continue
        last_row = prior.sort_values("game_date").iloc[-1]
        available = {
            c: last_row[c]
            for c in BATTER_AGG_FEATURES
            if c in last_row.index and pd.notna(last_row[c])
        }
        if available:
            lineup_feats.append(available)

    if not lineup_feats:
        return {"lineup_n_batters": 0}

    lf = pd.DataFrame(lineup_feats)
    result = {}
    for col in lf.columns:
        vals = lf[col].dropna()
        if vals.empty:
            continue
        result[f"lineup_mean_{col}"] = float(vals.mean())
        result[f"lineup_max_{col}"] = float(vals.max())
        if len(vals) >= 3:
            result[f"lineup_std_{col}"] = float(vals.std())

    result["lineup_n_batters"] = len(lineup_feats)
    result["lineup_n_with_bvp"] = int(lf["bvp_has_history"].sum()) if "bvp_has_history" in lf.columns else 0
    result["lineup_mean_bvp_pa"] = float(lf["bvp_pa_count"].mean()) if "bvp_pa_count" in lf.columns else 0.0
    return result


def build_feature_vector_for_starter(
    pitcher_id: int,
    lineup: list[int],
    feat_cols: list[str],
    profiles: pd.DataFrame,
    matchup_lookup: dict[tuple[int, int], dict] | None = None,
    pa_features: pd.DataFrame | None = None,
    game_date: pd.Timestamp | None = None,
) -> pd.DataFrame | None:
    """Build game-level feature vector for one starter."""
    pitcher_feats = _get_pitcher_features(pitcher_id, profiles)

    if pa_features is not None and game_date is not None:
        lineup_feats = _aggregate_lineup_from_pa_features(lineup, pa_features, game_date)
    elif matchup_lookup is not None:
        lineup_feats = _aggregate_lineup_from_matchups(lineup, pitcher_id, matchup_lookup)
    else:
        lineup_feats = {"lineup_n_batters": 0}

    today = pd.Timestamp.now()
    context = {
        "month": today.month,
        "day_of_week": today.dayofweek,
        "is_home": 0,
    }

    combined = {**pitcher_feats, **lineup_feats, **context}

    row = {c: combined.get(c, np.nan) for c in feat_cols}
    return pd.DataFrame([row])[feat_cols]


def predict_starter_runs(
    game_results: list[dict],
    pa_map: dict[int, int],
    coverage: str = DEFAULT_COVERAGE,
) -> list[dict]:
    """Predict runs for each starter using the trained model + conformal intervals."""
    model = load_starter_runs_model()
    conformal = load_conformal_quantiles()
    feat_cols = load_feature_cols()
    profiles = load_pitcher_profiles()

    q = conformal.get(coverage, conformal.get("80"))
    q_lo, q_hi = q["q_lo"], q["q_hi"]

    pa_feat_path = MASTER_DIR / "features_val_league.parquet"
    pa_features = None
    if pa_feat_path.exists():
        pa_features = pd.read_parquet(pa_feat_path)
        pa_features["game_date"] = pd.to_datetime(pa_features["game_date"])

    rows = []
    for gr in game_results:
        g = gr["game"]
        matchups = gr.get("matchups", [])
        lookup = {
            (int(m["batter_mlbam_id"]), int(m["pitcher_mlbam_id"])): m
            for m in matchups
        } if matchups else {}

        away_pid = int(g.get("away_pitcher_id") or 0)
        home_pid = int(g.get("home_pitcher_id") or 0)
        home_roster = [int(b["mlbam_id"]) for b in g.get("home_batters") or []]
        away_roster = [int(b["mlbam_id"]) for b in g.get("away_batters") or []]
        home_lineup = top9_lineup(home_roster, pa_map)
        away_lineup = top9_lineup(away_roster, pa_map)

        game_date = pd.Timestamp(g.get("game_date", pd.Timestamp.now()))

        result = {
            "game_pk": g.get("game_pk"),
            "game_date": str(g.get("game_date")),
            "matchup": f"{g.get('away_team')} @ {g.get('home_team')}",
        }

        for side, pid, lineup, is_home_flag in [
            ("away", away_pid, home_lineup, 0),
            ("home", home_pid, away_lineup, 1),
        ]:
            prefix = side
            result[f"{prefix}_pitcher"] = g.get(f"{side}_pitcher_name")
            result[f"{prefix}_pitcher_id"] = pid

            if pid <= 0:
                result[f"{prefix}_predicted_runs"] = None
                result[f"{prefix}_interval_low"] = None
                result[f"{prefix}_interval_high"] = None
                result[f"{prefix}_interval_width"] = None
                result[f"{prefix}_confidence"] = None
                continue

            X = build_feature_vector_for_starter(
                pitcher_id=pid,
                lineup=lineup,
                feat_cols=feat_cols,
                profiles=profiles,
                matchup_lookup=lookup,
                pa_features=pa_features,
                game_date=game_date,
            )

            if X is not None:
                X = X.fillna(0)
                if is_home_flag:
                    if "is_home" in X.columns:
                        X["is_home"] = 1
                pred = float(np.clip(model.predict(X)[0], 0, None))
            else:
                pred = 2.5

            lo = round(max(0, pred + q_lo), 2)
            hi = round(pred + q_hi, 2)
            width = round(hi - lo, 2)

            result[f"{prefix}_predicted_runs"] = round(pred, 2)
            result[f"{prefix}_interval_low"] = lo
            result[f"{prefix}_interval_high"] = hi
            result[f"{prefix}_interval_width"] = width
            result[f"{prefix}_confidence"] = _confidence_label(width)

        rows.append(result)
    return rows


def main():
    from narrative_engine import predict_matchups

    print("Running matchup predictions for lineup data...")
    game_results = predict_matchups()
    pa_map = load_qualifying_pa_map()

    print("Predicting starter runs...")
    table = predict_starter_runs(game_results, pa_map, coverage=DEFAULT_COVERAGE)

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    out = REPORTS_DIR / "todays_starter_run_expectancies.json"
    with open(out, "w") as f:
        json.dump(table, f, indent=2)
    print(f"Wrote {len(table)} games → {out}")

    for row in table:
        print(f"\n  {row['matchup']}")
        for side in ["away", "home"]:
            name = row.get(f"{side}_pitcher", "TBD")
            pred = row.get(f"{side}_predicted_runs")
            lo = row.get(f"{side}_interval_low")
            hi = row.get(f"{side}_interval_high")
            conf = row.get(f"{side}_confidence")
            if pred is not None:
                print(f"    {name:25s}  {pred:.2f} runs  [{lo:.1f} – {hi:.1f}]  {conf}")
            else:
                print(f"    {name:25s}  TBD")

    return table


if __name__ == "__main__":
    main()
