"""
Feature engineering pipeline for the PA-level ML model.

Takes raw PA-level Statcast data and builds shifted rolling features
to predict hit probability. Strict temporal discipline:
  - All rolling features use ONLY prior PAs (shifted by 1)
  - Train: game_year 2015–2024
  - Val:   game_year 2025

Output: data/master/features_train_league.parquet, data/master/features_val_league.parquet

League builds also emit **xBABIP-style** columns: shifted rolling means of
``estimated_ba_using_speedangle`` and ``estimated_woba_using_speedangle`` on
non-HR BIP (tracked ``launch_speed``), windows 30/100 — ``roll_est_*_bip_*`` and
``p_roll_est_*_bip_*``.
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from config import RAW_DIR, MASTER_DIR

ROOT = Path(__file__).resolve().parents[1]  # used by add_park_features

PA_PATH = RAW_DIR / "statcast_pa_level.parquet"
HIST_SPLITS_PATH = RAW_DIR / "historical_splits.parquet"

PA_PATH_LEAGUE = RAW_DIR / "statcast_pa_level_league.parquet"
HIST_SPLITS_PATH_LEAGUE = RAW_DIR / "historical_splits_league.parquet"

PITCHER_PROFILES_PATH = RAW_DIR / "pitcher_profiles_by_season.parquet"
BATTER_PITCH_PROFILES_PATH = RAW_DIR / "batter_pitch_profiles.parquet"
BVP_PATH = RAW_DIR / "bvp_matchup_features.parquet"

# Temporal decay for career platoon splits (season-level): weight season s at
# λ^((game_year - 1) - s) so the most recent full season before ``game_year`` has weight 1.0.
PLATOON_DECAY_LAMBDA = 0.7

ROLLING_WINDOWS = [10, 30, 100]
# xBABIP-style: shifted rolling mean of Statcast expected stats on non-HR BIP only.
# Longer windows only (skip 10) — high variance on BIP subsamples.
XBABIP_BIP_ROLL_WINDOWS = (30, 100)
# Very short windows (L3 / L5). L10 is already covered by ROLLING_WINDOWS.
SHORT_GAME_ROLLING_WINDOWS = (3, 5)
# True last-K-**games** rolls (shifted); used with recency league build.
GAME_ROLL_WINDOWS = (3, 5, 10)


def load_pa_data(league: bool = False) -> pd.DataFrame:
    path = PA_PATH_LEAGUE if league else PA_PATH
    df = pd.read_parquet(path)
    df["game_date"] = pd.to_datetime(df["game_date"])
    if "is_xbh" not in df.columns and "events" in df.columns:
        ev = df["events"].fillna("").str.lower()
        df["is_xbh"] = ev.isin({"double", "triple", "home_run"}).astype(int)
    df = df.sort_values(["batter", "game_date", "at_bat_number"]).reset_index(drop=True)
    return df


def add_rolling_features(
    df: pd.DataFrame,
    *,
    include_est_woba: bool = False,
) -> pd.DataFrame:
    """Add shifted rolling stats per batter. All windows use .shift(1) to prevent leakage."""
    df = df.copy()
    min_p = lambda w: max(1, w // 5)

    for w in ROLLING_WINDOWS:
        suffix = f"_{w}"
        grp = df.groupby("batter")

        df[f"roll_ba{suffix}"] = grp["is_hit"].transform(
            lambda x: x.shift(1).rolling(w, min_periods=min_p(w)).mean()
        )
        df[f"roll_hr_rate{suffix}"] = grp["is_hr"].transform(
            lambda x: x.shift(1).rolling(w, min_periods=min_p(w)).mean()
        )
        df[f"roll_k_rate{suffix}"] = grp["is_strikeout"].transform(
            lambda x: x.shift(1).rolling(w, min_periods=min_p(w)).mean()
        )
        df[f"roll_bb_rate{suffix}"] = grp["is_walk"].transform(
            lambda x: x.shift(1).rolling(w, min_periods=min_p(w)).mean()
        )

        if "launch_speed" in df.columns:
            df[f"roll_ev{suffix}"] = grp["launch_speed"].transform(
                lambda x: x.shift(1).rolling(w, min_periods=min_p(w)).mean()
            )
        if "launch_angle" in df.columns:
            df[f"roll_la{suffix}"] = grp["launch_angle"].transform(
                lambda x: x.shift(1).rolling(w, min_periods=min_p(w)).mean()
            )
        if "barrel" in df.columns:
            df[f"roll_barrel{suffix}"] = grp["barrel"].transform(
                lambda x: x.shift(1).rolling(w, min_periods=min_p(w)).mean()
            )
        if include_est_woba and "estimated_woba_using_speedangle" in df.columns:
            df[f"roll_est_woba{suffix}"] = grp["estimated_woba_using_speedangle"].transform(
                lambda x: x.shift(1).rolling(w, min_periods=min_p(w)).mean()
            )

    return df


def _shift_roll_mean(series: pd.Series, window: int) -> pd.Series:
    """Prior-PAs-only rolling mean (shift then roll)."""
    mp = max(1, window // 5)
    return series.shift(1).rolling(window, min_periods=mp).mean()


def add_xbabip_style_bip_roll_features(df: pd.DataFrame) -> pd.DataFrame:
    """Shifted rolling mean of Statcast ``estimated_*`` on non-HR balls in play.

    BIP mask: ``launch_speed`` present (tracked batted ball) and event is not
    ``home_run`` — aligns with Statcast expected-BA coverage on playable BIP.

    Adds batter rolls ``roll_est_ba_bip_{30,100}``, ``roll_est_woba_bip_{30,100}``
    and pitcher-allowed mirrors ``p_roll_est_*`` (same windows).

    No temporal leakage: all rolls use ``shift(1)`` before ``rolling``.
    """
    df = df.copy()
    if "batter" not in df.columns or "pitcher" not in df.columns:
        return df
    if "events" not in df.columns:
        return df
    ev = df["events"].fillna("").str.lower()
    is_hr = ev == "home_run"
    if "launch_speed" not in df.columns:
        return df
    ls = pd.to_numeric(df["launch_speed"], errors="coerce")
    bip_mask = ls.notna() & (~is_hr)

    tmp_cols: list[str] = []
    has_ba = "estimated_ba_using_speedangle" in df.columns
    has_xw = "estimated_woba_using_speedangle" in df.columns
    if not has_ba and not has_xw:
        return df

    grp_b = df.groupby("batter", sort=False)
    grp_p = df.groupby("pitcher", sort=False)

    for w in XBABIP_BIP_ROLL_WINDOWS:
        if has_ba:
            sba = pd.to_numeric(df["estimated_ba_using_speedangle"], errors="coerce")
            col_ba = f"_xbip_ba_{w}"
            df[col_ba] = sba.where(bip_mask)
            tmp_cols.append(col_ba)
            df[f"roll_est_ba_bip_{w}"] = grp_b[col_ba].transform(
                lambda s, ww=w: _shift_roll_mean(s, ww)
            )
            df[f"p_roll_est_ba_bip_{w}"] = grp_p[col_ba].transform(
                lambda s, ww=w: _shift_roll_mean(s, ww)
            )
        if has_xw:
            sxw = pd.to_numeric(df["estimated_woba_using_speedangle"], errors="coerce")
            col_xw = f"_xbip_xw_{w}"
            df[col_xw] = sxw.where(bip_mask)
            tmp_cols.append(col_xw)
            df[f"roll_est_woba_bip_{w}"] = grp_b[col_xw].transform(
                lambda s, ww=w: _shift_roll_mean(s, ww)
            )
            df[f"p_roll_est_woba_bip_{w}"] = grp_p[col_xw].transform(
                lambda s, ww=w: _shift_roll_mean(s, ww)
            )

    if tmp_cols:
        df = df.drop(columns=[c for c in tmp_cols if c in df.columns])
    return df


def add_short_game_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    """Shifted L3 / L5 batter rolling rates (L10 remains in add_rolling_features)."""
    df = df.copy()
    for w in SHORT_GAME_ROLLING_WINDOWS:
        suffix = f"_{w}"
        min_p = max(1, (w + 1) // 2)
        grp = df.groupby("batter")

        df[f"roll_ba{suffix}"] = grp["is_hit"].transform(
            lambda x: x.shift(1).rolling(w, min_periods=min_p).mean()
        )
        df[f"roll_hr_rate{suffix}"] = grp["is_hr"].transform(
            lambda x: x.shift(1).rolling(w, min_periods=min_p).mean()
        )
        df[f"roll_k_rate{suffix}"] = grp["is_strikeout"].transform(
            lambda x: x.shift(1).rolling(w, min_periods=min_p).mean()
        )
        df[f"roll_bb_rate{suffix}"] = grp["is_walk"].transform(
            lambda x: x.shift(1).rolling(w, min_periods=min_p).mean()
        )
        if "launch_speed" in df.columns:
            df[f"roll_ev{suffix}"] = grp["launch_speed"].transform(
                lambda x: x.shift(1).rolling(w, min_periods=min_p).mean()
            )
        if "launch_angle" in df.columns:
            df[f"roll_la{suffix}"] = grp["launch_angle"].transform(
                lambda x: x.shift(1).rolling(w, min_periods=min_p).mean()
            )
        if "barrel" in df.columns:
            df[f"roll_barrel{suffix}"] = grp["barrel"].transform(
                lambda x: x.shift(1).rolling(w, min_periods=min_p).mean()
            )
    return df


def add_game_level_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    """Per-batter last-K-games hit/HR/XBH rates (prior games only), merged to each PA."""
    df = df.copy()
    req = {"batter", "game_pk", "game_date", "is_hit", "is_hr", "is_xbh"}
    if not req <= set(df.columns):
        return df

    gs = (
        df.groupby(["batter", "game_pk"], as_index=False)
        .agg(
            game_date=("game_date", "min"),
            g_pa=("is_hit", "count"),
            g_hits=("is_hit", "sum"),
            g_hr=("is_hr", "sum"),
            g_xbh=("is_xbh", "sum"),
        )
    )
    gs = gs.sort_values(["batter", "game_date", "game_pk"])
    gs["g_hit_rate"] = gs["g_hits"] / gs["g_pa"].clip(lower=1)
    gs["g_hr_rate"] = gs["g_hr"] / gs["g_pa"].clip(lower=1)
    gs["g_xbh_rate"] = gs["g_xbh"] / gs["g_pa"].clip(lower=1)

    for w in GAME_ROLL_WINDOWS:
        min_p = max(1, (w + 1) // 2)
        gs[f"g_roll_hit_rate_{w}"] = gs.groupby("batter", group_keys=False)["g_hit_rate"].transform(
            lambda x, ww=w, mp=min_p: x.shift(1).rolling(ww, min_periods=mp).mean()
        )
        gs[f"g_roll_hr_rate_{w}"] = gs.groupby("batter", group_keys=False)["g_hr_rate"].transform(
            lambda x, ww=w, mp=min_p: x.shift(1).rolling(ww, min_periods=mp).mean()
        )
        gs[f"g_roll_xbh_rate_{w}"] = gs.groupby("batter", group_keys=False)["g_xbh_rate"].transform(
            lambda x, ww=w, mp=min_p: x.shift(1).rolling(ww, min_periods=mp).mean()
        )

    merge_cols = ["batter", "game_pk"] + [c for c in gs.columns if c.startswith("g_roll_")]
    return df.merge(gs[merge_cols], on=["batter", "game_pk"], how="left")


def add_pitcher_game_level_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    """Per-pitcher last-K-games rates allowed (hits / HR / XBH per BF in prior games)."""
    df = df.copy()
    req = {"pitcher", "game_pk", "game_date", "is_hit", "is_hr", "is_xbh"}
    if not req <= set(df.columns):
        return df

    gs = (
        df.groupby(["pitcher", "game_pk"], as_index=False)
        .agg(
            game_date=("game_date", "min"),
            g_bf=("is_hit", "count"),
            g_hits_allowed=("is_hit", "sum"),
            g_hr_allowed=("is_hr", "sum"),
            g_xbh_allowed=("is_xbh", "sum"),
        )
    )
    gs = gs.sort_values(["pitcher", "game_date", "game_pk"])
    gs["g_hit_rate_allowed"] = gs["g_hits_allowed"] / gs["g_bf"].clip(lower=1)
    gs["g_hr_rate_allowed"] = gs["g_hr_allowed"] / gs["g_bf"].clip(lower=1)
    gs["g_xbh_rate_allowed"] = gs["g_xbh_allowed"] / gs["g_bf"].clip(lower=1)

    for w in GAME_ROLL_WINDOWS:
        min_p = max(1, (w + 1) // 2)
        gs[f"p_g_roll_hit_rate_{w}"] = gs.groupby("pitcher", group_keys=False)["g_hit_rate_allowed"].transform(
            lambda x, ww=w, mp=min_p: x.shift(1).rolling(ww, min_periods=mp).mean()
        )
        gs[f"p_g_roll_hr_rate_{w}"] = gs.groupby("pitcher", group_keys=False)["g_hr_rate_allowed"].transform(
            lambda x, ww=w, mp=min_p: x.shift(1).rolling(ww, min_periods=mp).mean()
        )
        gs[f"p_g_roll_xbh_rate_{w}"] = gs.groupby("pitcher", group_keys=False)["g_xbh_rate_allowed"].transform(
            lambda x, ww=w, mp=min_p: x.shift(1).rolling(ww, min_periods=mp).mean()
        )

    merge_cols = ["pitcher", "game_pk"] + [c for c in gs.columns if c.startswith("p_g_roll_")]
    return df.merge(gs[merge_cols], on=["pitcher", "game_pk"], how="left")


def add_pitcher_short_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    """Per-pitcher L3 / L5 rolling K/BB/hits allowed (shifted)."""
    if "pitcher" not in df.columns:
        return df
    df = df.copy()
    for w in SHORT_GAME_ROLLING_WINDOWS:
        suffix = f"_{w}"
        min_p = max(1, (w + 1) // 2)
        grp = df.groupby("pitcher")
        if "is_strikeout" in df.columns:
            df[f"p_roll_k{suffix}"] = grp["is_strikeout"].transform(
                lambda x: x.shift(1).rolling(w, min_periods=min_p).mean()
            )
        if "is_walk" in df.columns:
            df[f"p_roll_bb{suffix}"] = grp["is_walk"].transform(
                lambda x: x.shift(1).rolling(w, min_periods=min_p).mean()
            )
        if "is_hit" in df.columns:
            df[f"p_roll_hit_allowed{suffix}"] = grp["is_hit"].transform(
                lambda x: x.shift(1).rolling(w, min_periods=min_p).mean()
            )
    return df


def add_beast_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    """First-wave Beast HR features: per-PA hard-hit / laser / air-ball / ISO rolls (batter + pitcher).

    All features are .shift(1) rolling means so each PA only sees prior PAs.
    NaN inputs (non-BIP PAs without a launch_speed/launch_angle) are treated as 0
    so the rolls measure rate-per-PA, blending contact frequency and quality.
    """
    df = df.copy()
    if "launch_speed" in df.columns:
        ls = df["launch_speed"].astype(float)
        df["_beast_hard_hit"] = (ls >= 95).fillna(False).astype(int)
        df["_beast_laser"] = (ls >= 100).fillna(False).astype(int)
    if "launch_angle" in df.columns:
        la = df["launch_angle"].astype(float)
        df["_beast_air"] = (la >= 25).fillna(False).astype(int)

    min_p = lambda w: max(1, w // 5)

    if "batter" in df.columns:
        grp_b = df.groupby("batter")
        for w in ROLLING_WINDOWS:
            sfx = f"_{w}"
            if "_beast_hard_hit" in df.columns:
                df[f"roll_hard_hit{sfx}"] = grp_b["_beast_hard_hit"].transform(
                    lambda x, ww=w: x.shift(1).rolling(ww, min_periods=min_p(ww)).mean()
                )
            if "_beast_laser" in df.columns:
                df[f"roll_laser{sfx}"] = grp_b["_beast_laser"].transform(
                    lambda x, ww=w: x.shift(1).rolling(ww, min_periods=min_p(ww)).mean()
                )
            if "_beast_air" in df.columns:
                df[f"roll_air{sfx}"] = grp_b["_beast_air"].transform(
                    lambda x, ww=w: x.shift(1).rolling(ww, min_periods=min_p(ww)).mean()
                )
            if "iso_value" in df.columns:
                df[f"roll_iso_value{sfx}"] = grp_b["iso_value"].transform(
                    lambda x, ww=w: x.shift(1).rolling(ww, min_periods=min_p(ww)).mean()
                )

    if "pitcher" in df.columns:
        grp_p = df.groupby("pitcher")
        for w in (30, 100):
            sfx = f"_{w}"
            if "_beast_hard_hit" in df.columns:
                df[f"p_roll_hard_hit_allowed{sfx}"] = grp_p["_beast_hard_hit"].transform(
                    lambda x, ww=w: x.shift(1).rolling(ww, min_periods=min_p(ww)).mean()
                )
            if "_beast_laser" in df.columns:
                df[f"p_roll_laser_allowed{sfx}"] = grp_p["_beast_laser"].transform(
                    lambda x, ww=w: x.shift(1).rolling(ww, min_periods=min_p(ww)).mean()
                )
            if "_beast_air" in df.columns:
                df[f"p_roll_air_allowed{sfx}"] = grp_p["_beast_air"].transform(
                    lambda x, ww=w: x.shift(1).rolling(ww, min_periods=min_p(ww)).mean()
                )
            if "iso_value" in df.columns:
                df[f"p_roll_iso_value_allowed{sfx}"] = grp_p["iso_value"].transform(
                    lambda x, ww=w: x.shift(1).rolling(ww, min_periods=min_p(ww)).mean()
                )

    drop_cols = [c for c in ("_beast_hard_hit", "_beast_laser", "_beast_air") if c in df.columns]
    if drop_cols:
        df = df.drop(columns=drop_cols)
    return df


def platoon_decayed_rates_for_batter_year(
    mlbam_id: int,
    game_year: int,
    splits: pd.DataFrame | None = None,
    *,
    decay_lambda: float = PLATOON_DECAY_LAMBDA,
) -> dict[str, float]:
    """Decay-weighted career platoon evidence through seasons strictly before ``game_year``.

    Returns cum_* columns expected by the GBDT plus ``platoon_eff_*`` counts for
    post-hoc Bayesian vs-hand posteriors (not all are in ``feature_columns.json``).
    """
    out: dict[str, float] = {}
    if splits is None:
        splits_path = HIST_SPLITS_PATH_LEAGUE
        if not splits_path.exists():
            splits_path = HIST_SPLITS_PATH
        if not splits_path.exists():
            for suff in ("lhp", "rhp"):
                out[f"cum_career_ba_vs_{suff}"] = 0.0
                out[f"cum_career_pa_vs_{suff}"] = 0.0
                out[f"cum_career_ops_vs_{suff}"] = 0.0
                out[f"cum_career_hr_vs_{suff}"] = 0.0
                out[f"platoon_eff_h_{suff}"] = 0.0
                out[f"platoon_eff_ab_{suff}"] = 0.0
                out[f"platoon_eff_pa_{suff}"] = 0.0
                out[f"platoon_eff_hr_{suff}"] = 0.0
                out[f"platoon_eff_xbh_{suff}"] = 0.0
                out[f"platoon_eff_so_{suff}"] = 0.0
            return out
        splits = pd.read_parquet(splits_path)

    lam = float(decay_lambda)
    gy = int(game_year)

    for hand_label, suff in (("vs LHP", "lhp"), ("vs RHP", "rhp")):
        g = splits[(splits["mlbam_id"] == mlbam_id) & (splits["split"] == hand_label) & (splits["season"] < gy)]
        if g.empty:
            out[f"cum_career_ba_vs_{suff}"] = 0.0
            out[f"cum_career_pa_vs_{suff}"] = 0.0
            out[f"cum_career_ops_vs_{suff}"] = 0.0
            out[f"cum_career_hr_vs_{suff}"] = 0.0
            out[f"platoon_eff_h_{suff}"] = 0.0
            out[f"platoon_eff_ab_{suff}"] = 0.0
            out[f"platoon_eff_pa_{suff}"] = 0.0
            out[f"platoon_eff_hr_{suff}"] = 0.0
            out[f"platoon_eff_xbh_{suff}"] = 0.0
            out[f"platoon_eff_so_{suff}"] = 0.0
            continue

        seas = g["season"].to_numpy(dtype=float)
        w = np.power(lam, (gy - 1) - seas)
        pa = g["PA"].to_numpy(dtype=float)
        ab = g["AB"].to_numpy(dtype=float) if "AB" in g.columns else pa
        h = g["H"].to_numpy(dtype=float)
        hr = g["HR"].to_numpy(dtype=float)
        d2 = g["2B"].to_numpy(dtype=float) if "2B" in g.columns else np.zeros(len(g))
        d3 = g["3B"].to_numpy(dtype=float) if "3B" in g.columns else np.zeros(len(g))
        so_col = g["SO"].to_numpy(dtype=float) if "SO" in g.columns else np.zeros(len(g))
        ops = g["OPS"].to_numpy(dtype=float) if "OPS" in g.columns else np.zeros(len(g))

        eff_pa = float((w * pa).sum())
        eff_ab = float((w * ab).sum())
        eff_h = float((w * h).sum())
        eff_hr = float((w * hr).sum())
        xbh_row = d2 + d3 + hr
        eff_xbh = float((w * xbh_row).sum())
        eff_so = float((w * so_col).sum())
        eff_ops_num = float((w * ops * pa).sum())
        eff_ops_den = float((w * pa).sum())

        out[f"platoon_eff_pa_{suff}"] = eff_pa
        out[f"platoon_eff_ab_{suff}"] = eff_ab
        out[f"platoon_eff_h_{suff}"] = eff_h
        out[f"platoon_eff_hr_{suff}"] = eff_hr
        out[f"platoon_eff_xbh_{suff}"] = eff_xbh
        out[f"platoon_eff_so_{suff}"] = eff_so

        out[f"cum_career_pa_vs_{suff}"] = eff_pa
        out[f"cum_career_ba_vs_{suff}"] = float(eff_h / eff_ab) if eff_ab > 1e-6 else 0.0
        out[f"cum_career_hr_vs_{suff}"] = float(eff_hr / eff_pa) if eff_pa > 1e-6 else 0.0
        out[f"cum_career_ops_vs_{suff}"] = float(eff_ops_num / eff_ops_den) if eff_ops_den > 1e-6 else 0.0

    return out


def add_career_split_features(df: pd.DataFrame, league: bool = False) -> pd.DataFrame:
    """Merge historical platoon split stats (PA-weighted, λ-decayed through prior seasons)."""
    splits_path = HIST_SPLITS_PATH_LEAGUE if league else HIST_SPLITS_PATH
    if not splits_path.exists():
        print(f"  No historical splits found at {splits_path}, skipping career features")
        return df

    splits = pd.read_parquet(splits_path)
    if "game_year" not in df.columns:
        print("  add_career_split_features: no game_year column, skipping")
        return df

    pairs = df[["batter", "game_year"]].drop_duplicates()
    recs: list[dict] = []
    for bid, gy in zip(pairs["batter"].astype(int), pairs["game_year"].astype(int)):
        row = platoon_decayed_rates_for_batter_year(int(bid), int(gy), splits)
        row["batter"] = int(bid)
        row["game_year"] = int(gy)
        recs.append(row)

    look = pd.DataFrame(recs)
    merge_cols = [c for c in look.columns if c not in ("batter", "game_year")]
    drop_me = [c for c in merge_cols if c in df.columns]
    if drop_me:
        df = df.drop(columns=drop_me, errors="ignore")

    n_before = len(df)
    df = df.merge(look, on=["batter", "game_year"], how="left")
    assert len(df) == n_before, (
        f"Career split merge inflated rows: {n_before:,} → {len(df):,}"
    )
    return df


def add_context_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add pitcher handedness, count-based, and seasonal features."""
    df = df.copy()

    if "p_throws" in df.columns:
        df["vs_lhp"] = (df["p_throws"] == "L").astype(int)

    if "game_date" in df.columns:
        df["month"] = df["game_date"].dt.month
        df["day_of_week"] = df["game_date"].dt.dayofweek
        # C4: cyclical month encoding lets tree-based models capture the season cycle better
        # than a single integer (e.g. month=12 and month=1 should behave similarly).
        import numpy as _np
        df["month_sin"] = _np.sin(2 * _np.pi * df["month"] / 12.0)
        df["month_cos"] = _np.cos(2 * _np.pi * df["month"] / 12.0)
        # Days into season — finer granularity than month buckets. Anchored at March 1.
        season_start = pd.to_datetime(df["game_date"].dt.year.astype(str) + "-03-01")
        df["days_into_season"] = (df["game_date"] - season_start).dt.days.clip(lower=0)

    if "pitch_number" in df.columns:
        df["pitch_count"] = df["pitch_number"]

    if "zone" in df.columns:
        df["in_zone"] = df["zone"].fillna(0).between(1, 9).astype(int)

    return df


def add_platoon_aligned_split_features(df: pd.DataFrame) -> pd.DataFrame:
    """Expose platoon-aligned cumulative splits for the PA's pitcher hand (vs_lhp).

    Lets trees emphasize career performance **against the arm handedness faced
    on this PA** beyond raw vs_lhp × cum_* interactions.
    """
    df = df.copy()
    if "vs_lhp" not in df.columns:
        return df

    vl = pd.to_numeric(df["vs_lhp"], errors="coerce").fillna(0.0).astype(float)
    mask_l = vl >= 0.5

    ba_l, ba_r = df.get("cum_career_ba_vs_lhp"), df.get("cum_career_ba_vs_rhp")
    if ba_l is not None and ba_r is not None:
        df["platoon_matched_ba"] = np.where(mask_l, ba_l, ba_r)
        df["platoon_other_ba"] = np.where(mask_l, ba_r, ba_l)

    ops_l, ops_r = df.get("cum_career_ops_vs_lhp"), df.get("cum_career_ops_vs_rhp")
    if ops_l is not None and ops_r is not None:
        df["platoon_matched_ops"] = np.where(mask_l, ops_l, ops_r)
        df["platoon_other_ops"] = np.where(mask_l, ops_r, ops_l)
        df["platoon_ops_gap"] = (
            pd.to_numeric(df["platoon_matched_ops"], errors="coerce").fillna(0)
            - pd.to_numeric(df["platoon_other_ops"], errors="coerce").fillna(0)
        )

    return df


def add_pitcher_profile_features(df: pd.DataFrame) -> pd.DataFrame:
    """Merge per-pitcher-per-season profile features (arsenal, velo, quality)."""
    if not PITCHER_PROFILES_PATH.exists():
        print("  No pitcher profiles found, skipping")
        return df

    profiles = pd.read_parquet(PITCHER_PROFILES_PATH)

    keep = ["pitcher", "season",
            "pct_fastball", "pct_breaking", "pct_offspeed",
            "velo_fastball", "velo_breaking", "velo_overall",
            "spin_fastball", "spin_breaking",
            "pfx_x_fastball", "pfx_z_fastball", "pfx_x_breaking", "pfx_z_breaking",
            "pct_in_zone", "whiff_rate", "k_rate", "bb_rate",
            "barrel_rate_allowed", "arm_angle", "extension"]
    keep = [c for c in keep if c in profiles.columns]

    cum_profiles = []
    for pid, grp in profiles.sort_values(["pitcher", "season"]).groupby("pitcher"):
        grp = grp.copy()
        for col in keep[2:]:
            grp[f"p_{col}"] = grp[col].expanding().mean().shift(1)
        cum_profiles.append(grp)

    if not cum_profiles:
        return df
    cum_df = pd.concat(cum_profiles)
    p_cols = [c for c in cum_df.columns if c.startswith("p_")]
    cum_df = cum_df[["pitcher", "season"] + p_cols]

    n_before = len(df)
    df = df.merge(cum_df, left_on=["pitcher", "game_year"],
                  right_on=["pitcher", "season"], how="left")
    assert len(df) == n_before, (
        f"Pitcher profile merge inflated rows: {n_before:,} → {len(df):,}"
    )
    df = df.drop(columns=["season"], errors="ignore")
    return df


def add_pitcher_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add shifted rolling stats per pitcher (K rate, BB rate, barrel%)."""
    if "pitcher" not in df.columns:
        return df

    df = df.copy()
    for w in [10, 30]:
        suffix = f"_{w}"
        grp = df.groupby("pitcher")
        if "is_strikeout" in df.columns:
            df[f"p_roll_k{suffix}"] = grp["is_strikeout"].transform(
                lambda x: x.shift(1).rolling(w, min_periods=max(1, w // 5)).mean()
            )
        if "is_walk" in df.columns:
            df[f"p_roll_bb{suffix}"] = grp["is_walk"].transform(
                lambda x: x.shift(1).rolling(w, min_periods=max(1, w // 5)).mean()
            )
        if "is_hit" in df.columns:
            df[f"p_roll_hit_allowed{suffix}"] = grp["is_hit"].transform(
                lambda x: x.shift(1).rolling(w, min_periods=max(1, w // 5)).mean()
            )
    return df


def add_batter_pitch_type_features(
    df: pd.DataFrame,
    *,
    include_barrel_bpt: bool = False,
) -> pd.DataFrame:
    """Merge batter's performance by pitch-type group (shifted by season)."""
    if not BATTER_PITCH_PROFILES_PATH.exists():
        print("  No batter pitch profiles found, skipping")
        return df

    bpp = pd.read_parquet(BATTER_PITCH_PROFILES_PATH)

    keep = ["batter", "season",
            "ba_vs_fastball", "ev_vs_fastball", "whiff_vs_fastball",
            "ba_vs_breaking", "ev_vs_breaking", "whiff_vs_breaking",
            "ba_vs_offspeed", "ev_vs_offspeed", "whiff_vs_offspeed",
            "ev_vs_lhp", "ev_vs_rhp"]
    if include_barrel_bpt:
        for _pg in ("fastball", "breaking", "offspeed"):
            c = f"barrel_vs_{_pg}"
            if c in bpp.columns:
                keep.append(c)
    keep = [c for c in keep if c in bpp.columns]

    cum_bpp = []
    for bid, grp in bpp.sort_values(["batter", "season"]).groupby("batter"):
        grp = grp.copy()
        for col in keep[2:]:
            grp[f"bpt_{col}"] = grp[col].expanding().mean().shift(1)
        cum_bpp.append(grp)

    if not cum_bpp:
        return df
    cum_df = pd.concat(cum_bpp)
    bpt_cols = [c for c in cum_df.columns if c.startswith("bpt_")]
    cum_df = cum_df[["batter", "season"] + bpt_cols]

    n_before = len(df)
    df = df.merge(cum_df, left_on=["batter", "game_year"],
                  right_on=["batter", "season"], how="left")
    assert len(df) == n_before, (
        f"Batter pitch-type merge inflated rows: {n_before:,} → {len(df):,}"
    )
    df = df.drop(columns=["season"], errors="ignore")
    return df


def add_bvp_features(df: pd.DataFrame) -> pd.DataFrame:
    """Merge BvP matchup features (cumulative shifted stats per batter-pitcher pair)."""
    if not BVP_PATH.exists():
        print("  No BvP features found, skipping")
        return df

    bvp = pd.read_parquet(BVP_PATH)
    df = df.merge(bvp, left_index=True, right_index=True, how="left")
    return df


def add_park_features(df: pd.DataFrame) -> pd.DataFrame:
    """L2: join two-sided park HR factor + enclosure one-hots per game.

    Uses data/priors/park_hr_factors.json (built by src/build_park_hr_factors.py)
    and data/raw/park_lookup.csv. Requires `home_team` (joined from bulk Statcast
    by game_pk if not already present).
    """
    import json as _json
    pf_path = ROOT / "data" / "priors" / "park_hr_factors.json"
    if not pf_path.exists():
        print("  No park_hr_factors.json found; run src/build_park_hr_factors.py")
        return df

    df = df.copy()
    if "home_team" not in df.columns:
        bulk_dir = RAW_DIR / "statcast_bulk"
        years_in_data = sorted(pd.to_datetime(df["game_date"]).dt.year.dropna().unique().astype(int)) \
            if "game_date" in df.columns else []
        gp_map = {}
        for y in years_in_data:
            p = bulk_dir / f"statcast_{y}.parquet"
            if not p.exists():
                continue
            bulk = pd.read_parquet(p, columns=["game_pk", "home_team"]).drop_duplicates("game_pk")
            for gp, ht in zip(bulk["game_pk"], bulk["home_team"]):
                gp_map[gp] = ht
        if not gp_map:
            print("  Cannot resolve home_team without bulk Statcast; skip park features")
            return df
        df["home_team"] = df["game_pk"].map(gp_map)

    pf = _json.loads(pf_path.read_text()).get("parks", {})
    df["park_pf_hr"] = df["home_team"].map(
        lambda ht: pf.get(str(ht), {}).get("pf_two_sided_shrunk", 1.0)
    )

    park_lookup_path = RAW_DIR / "park_lookup.csv"
    if park_lookup_path.exists():
        park_lookup = pd.read_csv(park_lookup_path)
        enc_map = dict(zip(park_lookup["home_team"], park_lookup["enclosure"]))
        df["park_enclosure"] = df["home_team"].map(enc_map).fillna("outdoor")
        df["park_enclosure_outdoor"] = (df["park_enclosure"] == "outdoor").astype(int)
        df["park_enclosure_dome"] = (df["park_enclosure"] == "fixed_dome").astype(int)
        df["park_enclosure_retractable"] = (df["park_enclosure"] == "retractable").astype(int)
        df = df.drop(columns=["park_enclosure"], errors="ignore")

    n_park = df["park_pf_hr"].notna().sum()
    print(f"  Park features joined for {n_park:,} / {len(df):,} rows "
          f"({100*n_park/max(len(df),1):.1f}%)")
    return df


def add_pitcher_hr_profile_features(df: pd.DataFrame) -> pd.DataFrame:
    """T2.2: per-pitcher HR profile features (LA / EV / pull% / zone% / count adv%).

    Reads data/raw/pitcher_hr_profile.parquet (built by src/build_pitcher_hr_profile.py)
    and left-joins on (pitcher, season). Features:
      p_hr_la_mean, p_hr_ev_mean, p_hr_pulled_pct, p_hr_zone_rate, p_hr_count_advantage_pct
    """
    p = RAW_DIR / "pitcher_hr_profile.parquet"
    if not p.exists():
        print("  No pitcher_hr_profile.parquet (run src/build_pitcher_hr_profile.py); skipping")
        return df
    prof = pd.read_parquet(p)
    df = df.copy()
    if "season" not in df.columns and "game_date" in df.columns:
        df["season"] = pd.to_datetime(df["game_date"]).dt.year
    if "pitcher" not in df.columns or "season" not in df.columns:
        print("  Pitcher HR profile join skipped (missing pitcher/season cols)")
        return df
    n_before = df["pitcher"].notna().sum()
    df = df.merge(prof, on=["pitcher", "season"], how="left",
                  suffixes=("", "_pitcher_hr_drop"))
    # Drop the duplicate n_hr_allowed column created by the merge if it conflicts
    df = df.drop(columns=[c for c in df.columns if c.endswith("_pitcher_hr_drop")], errors="ignore")
    n_join = df["p_hr_la_mean"].notna().sum() if "p_hr_la_mean" in df.columns else 0
    print(f"  Pitcher HR profile coverage: {n_join:,} / {n_before:,} rows "
          f"({100*n_join/max(n_before,1):.1f}%)")
    return df


def add_park_arsenal_interactions(df: pd.DataFrame) -> pd.DataFrame:
    """T2.4: hand-engineered park × pitcher arsenal interactions.

    Trees can theoretically learn these from `park_pf_hr` and `p_pct_*` separately,
    but explicit interactions improve generalization on rare combinations. Three
    new features:
      park_x_fastball, park_x_breaking, park_x_offspeed
    """
    df = df.copy()
    if "park_pf_hr" not in df.columns:
        return df
    for pt in ("fastball", "breaking", "offspeed"):
        col = f"p_pct_{pt}"
        if col in df.columns:
            df[f"park_x_{pt}"] = df["park_pf_hr"].fillna(1.0) * df[col].fillna(0.0)
    return df


def add_xptw_features(df: pd.DataFrame) -> pd.DataFrame:
    """M4: pitch-type-weighted batter features per (batter, pitcher, season).

    Reads data/raw/xptw_features.parquet (built by src/build_xptw_features.py)
    and left-joins on (batter, pitcher, season). Features:
      xptw_p_hr_rate (T2.1), xptw_p_barrel, xptw_p_ba, xptw_p_ev, xptw_p_whiff
    Missing pairs (no profile data for either side) get NaN, handled by
    SimpleImputer at training time.
    """
    xptw_path = RAW_DIR / "xptw_features.parquet"
    if not xptw_path.exists():
        print("  No xptw features found (run src/build_xptw_features.py); skipping")
        return df

    xptw = pd.read_parquet(xptw_path)
    df = df.copy()
    if "season" not in df.columns and "game_date" in df.columns:
        df["season"] = pd.to_datetime(df["game_date"]).dt.year
    if "season" not in df.columns or "batter" not in df.columns or "pitcher" not in df.columns:
        print("  xptw join skipped — missing batter/pitcher/season key cols")
        return df
    n_before = df["pitcher"].notna().sum()
    df = df.merge(xptw, on=["batter", "pitcher", "season"], how="left")
    n_xptw = df["xptw_p_barrel"].notna().sum() if "xptw_p_barrel" in df.columns else 0
    print(f"  xptw join coverage: {n_xptw:,} / {n_before:,} rows ({100*n_xptw/max(n_before,1):.1f}%)")
    return df


def add_pitcher_context_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add pitcher context features from the PA-level data."""
    df = df.copy()
    if "n_thruorder_pitcher" in df.columns:
        df["times_thru_order"] = df["n_thruorder_pitcher"].fillna(1).astype(int)
    if "pitcher_days_since_prev_game" in df.columns:
        df["pitcher_rest_days"] = df["pitcher_days_since_prev_game"].fillna(5).clip(0, 30)
    if "age_pit" in df.columns:
        df["pitcher_age"] = df["age_pit"].fillna(28)
    return df


def build_feature_set(
    league: bool = False,
    *,
    include_contact_experiment: bool = False,
    include_short_game_rolls: bool = False,
    include_beast_rolls: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Full pipeline: load PA data → engineer features → split train/val.

    When ``include_contact_experiment`` is True (league builds with ``--exp``),
    adds ``roll_est_woba_*`` and ``bpt_barrel_vs_*`` for the experiment model bundle.
    Default False preserves the legacy feature set for production parquet.

    When ``include_short_game_rolls`` is True (``--recency`` league build), adds
    L3/L5 PA-window rolls, true last-K-**games** rolls (``g_roll_*``, ``p_g_roll_*``),
    and pitcher L3/L5 PA rolls (L10/30/100 PA windows already exist).

    When ``include_beast_rolls`` is True (``--beast`` league build), inherits the
    ``--exp`` superset and additionally adds first-wave HR features:
    per-PA hard-hit (EV≥95), laser (EV≥100), air-ball (LA≥25), and ``iso_value``
    rolling rates for both batter and pitcher.
    """
    mode = "LEAGUE-WIDE" if league else "14-player"
    print(f"Loading PA-level data ({mode})...")
    df = load_pa_data(league=league)
    print(f"  {len(df):,} PAs loaded")

    print("Adding rolling features...")
    df = add_rolling_features(
        df,
        include_est_woba=include_contact_experiment or include_beast_rolls,
    )

    print("Adding xBABIP-style BIP rolling features (est BA / xwoba on non-HR BIP)...")
    df = add_xbabip_style_bip_roll_features(df)

    if include_beast_rolls:
        print("Adding Beast first-wave HR rolling features (hard-hit / laser / air / iso)...")
        df = add_beast_rolling_features(df)

    if include_short_game_rolls:
        print("Adding L3/L5 short-window rolling features...")
        df = add_short_game_rolling_features(df)

    print("Adding career split features...")
    df = add_career_split_features(df, league=league)

    print("Adding context features...")
    df = add_context_features(df)
    print("Adding platoon-aligned split features...")
    df = add_platoon_aligned_split_features(df)

    if league:
        print("Adding pitcher profile features...")
        df = add_pitcher_profile_features(df)

        print("Adding pitcher rolling features...")
        df = add_pitcher_rolling_features(df)
        if include_short_game_rolls:
            print("Adding pitcher L3/L5 rolling features...")
            df = add_pitcher_short_rolling_features(df)
            print("Adding game-level L3/L5/L10 rolling features (prior games)...")
            df = add_game_level_rolling_features(df)
            df = add_pitcher_game_level_rolling_features(df)

        print("Adding batter pitch-type features...")
        df = add_batter_pitch_type_features(
            df, include_barrel_bpt=include_contact_experiment or include_beast_rolls
        )

        print("Adding pitcher context features...")
        df = add_pitcher_context_features(df)

        print("Adding BvP matchup features...")
        df = add_bvp_features(df)

        print("Adding xptw (M4) features...")
        df = add_xptw_features(df)

        print("Adding park (L2) features...")
        df = add_park_features(df)

        print("Adding pitcher HR profile (T2.2) features...")
        df = add_pitcher_hr_profile_features(df)

        print("Adding park × arsenal (T2.4) interactions...")
        df = add_park_arsenal_interactions(df)

    feature_prefixes = ("roll_", "cum_", "vs_lhp", "month", "day_of_week",
                        "days_into_season",
                        "pitch_count", "in_zone",
                        "platoon_",
                        "p_", "g_roll_", "p_g_roll_", "bpt_", "bvp_", "log_bvp_",
                        "xptw_", "park_", "park_x_", "p_hr_", "times_thru",
                        "pitcher_rest", "pitcher_age")
    non_numeric_cols = {"p_throws"}
    feature_cols = [c for c in df.columns
                    if c.startswith(feature_prefixes) and c not in non_numeric_cols]
    target_cols = ["is_hit", "is_hr", "is_strikeout", "is_walk", "is_ab", "is_xbh"]
    id_cols = ["batter", "pitcher", "game_pk", "game_date", "game_year", "at_bat_number",
               "events", "player_name_clean", "home_team", "venue"]
    available_id = [c for c in id_cols if c in df.columns]
    available_feat = [c for c in feature_cols if c in df.columns]
    available_tgt = [c for c in target_cols if c in df.columns]

    keep = available_id + available_feat + available_tgt
    df = df[[c for c in keep if c in df.columns]]

    train = df[df["game_year"] <= 2024].copy()
    val = df[df["game_year"] >= 2025].copy()

    val_years = sorted(val["game_year"].unique().tolist()) if not val.empty else []
    print(f"\nFeature columns: {len(available_feat)}")
    print(f"Train PAs: {len(train):,} (2015–2024)")
    print(f"Val PAs:   {len(val):,} ({val_years if val_years else 'empty'})")

    null_pct = train[available_feat].isnull().mean()
    high_null = null_pct[null_pct > 0.5]
    if not high_null.empty:
        print(f"\nWarning: features with >50% null in train:")
        print(high_null.to_string())

    return train, val


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--league", action="store_true", help="Use league-wide data")
    parser.add_argument(
        "--exp",
        action="store_true",
        help="League only: add roll_est_woba_* + bpt_barrel_vs_*; write features_*_league_exp.parquet",
    )
    parser.add_argument(
        "--recency",
        action="store_true",
        help="League only: add L3/L5 batter+pitcher rolls; write features_*_league_recency.parquet",
    )
    parser.add_argument(
        "--beast",
        action="store_true",
        help="League only: --exp superset + first-wave HR rolls; write features_*_league_beast.parquet",
    )
    args = parser.parse_args()

    if args.exp and not args.league:
        raise SystemExit("--exp requires --league")
    if args.recency and not args.league:
        raise SystemExit("--recency requires --league")
    if args.beast and not args.league:
        raise SystemExit("--beast requires --league")
    if sum(int(x) for x in (args.exp, args.recency, args.beast)) > 1:
        raise SystemExit("Use only one of --exp / --recency / --beast per run.")

    suffix = "_league" if args.league else ""
    if args.exp:
        suffix = "_league_exp"
    if args.recency:
        suffix = "_league_recency"
    if args.beast:
        suffix = "_league_beast"
    mode = "LEAGUE-WIDE" if args.league else "14-PLAYER"
    if args.exp:
        mode += " + CONTACT-EXPERIMENT"
    if args.recency:
        mode += " + L3/L5 RECENCY ROLLS"
    if args.beast:
        mode += " + BEAST HR FIRST-WAVE"

    print("=" * 60)
    print(f"  FEATURE ENGINEERING PIPELINE ({mode})")
    print("=" * 60)

    train, val = build_feature_set(
        league=args.league,
        include_contact_experiment=args.exp,
        include_short_game_rolls=args.recency,
        include_beast_rolls=args.beast,
    )

    train.to_parquet(MASTER_DIR / f"features_train{suffix}.parquet", index=False)
    val.to_parquet(MASTER_DIR / f"features_val{suffix}.parquet", index=False)
    print(f"\nSaved train → features_train{suffix}.parquet")
    print(f"Saved val   → features_val{suffix}.parquet")

    feature_cols = [c for c in train.columns if c.startswith(("roll_", "cum_", "vs_lhp", "month",
                                                                "day_of_week", "days_into_season",
                                                                "pitch_count", "in_zone",
                                                                "platoon_",
                                                                "p_", "g_roll_", "p_g_roll_", "bpt_", "bvp_", "log_bvp_",
                                                                "xptw_", "park_", "park_x_", "p_hr_",
                                                                "times_thru", "pitcher_rest", "pitcher_age"))]
    print(f"\nFeatures ({len(feature_cols)}):")
    for c in sorted(feature_cols):
        print(f"  {c}")

    print(f"\nTrain hit rate: {train['is_hit'].mean():.3f}")
    print(f"Val hit rate:   {val['is_hit'].mean():.3f}")
