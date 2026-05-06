"""
Residual longshots: rank production HR matchups by lift vs a conservative prior,
with obviousness (slate rank + YTD + career) so the list favors non-obvious names.

Used at dashboard generation time only (no change to prediction models).
Data: todays_matchup_predictions.json rows + qualifying_batters_*.csv for YTD HR/PA.

evaluate_residual_history (read-only) joins archived production prediction snapshots
with Statcast PA-level outcomes for the slate calendar day, computing:
  - Tier A: HR vs the listed starter on slate_date (strict matchup grain)
  - Tier B: any HR on slate_date (lenient batter-day grain)
against the residual list and an in-band baseline (rows in [ADJ_HR_MIN, ADJ_HR_MAX])
so the residual longshots tab can show whether the screen is paying or not.
"""
from __future__ import annotations

import csv
import json
import re
from dataclasses import dataclass
from pathlib import Path

# Pinned defaults — tune with `src/tune_residual_longshots.py` + archive + Statcast; re-run as slates accrue.
# Grid 2026-04-27 (7 slates, Tier A: HR vs listed starter, vs in-band [ADJ_HR_MIN, ADJ_HR_MAX] baseline):
#   best d_tier_a +0.45 pp, d_tier_b +0.57 pp, n_residual=95 — small sample, not stable proof.
LEAGUE_HR_PRIOR = 0.031
PRIOR_FLOOR = 0.004
ADJ_HR_MIN = 0.010
ADJ_HR_MAX = 0.058
MAX_SLATE_PCT = 0.88  # was 0.92; exclude top ~12% of slate (model-obvious)
MIN_LIFT = 1.01
MIN_CONF_HR = 0.65
MIN_PA_FOR_YTD = 12
BEAST_HR_DIVERGENCE = 0.015
W_SLATE = 0.35
W_YTD = 0.15
W_CAREER = 0.50
TOP_N = 25


@dataclass(frozen=True)
class ResidualConfig:
    """Tunable screen parameters; defaults match module constants."""

    adj_hr_min: float = ADJ_HR_MIN
    adj_hr_max: float = ADJ_HR_MAX
    max_slate_pct: float = MAX_SLATE_PCT
    min_lift: float = MIN_LIFT
    min_conf_hr: float = MIN_CONF_HR
    min_pa_for_ytd: int = MIN_PA_FOR_YTD
    league_hr_prior: float = LEAGUE_HR_PRIOR
    prior_floor: float = PRIOR_FLOOR
    w_slate: float = W_SLATE
    w_ytd: float = W_YTD
    w_career: float = W_CAREER
    beast_hr_divergence: float = BEAST_HR_DIVERGENCE
    top_n: int = TOP_N

    @classmethod
    def defaults(cls) -> ResidualConfig:
        return cls()


def _f(x, default: float = 0.0) -> float:
    try:
        return float(x)
    except (TypeError, ValueError):
        return default


def _percentile_rank(sorted_vals: list[float], v: float) -> float:
    """Fraction of values <= v (inclusive), in [0, 1]."""
    if not sorted_vals:
        return 0.5
    n = len(sorted_vals)
    le = 0
    for x in sorted_vals:
        if x <= v:
            le += 1
        else:
            break
    return le / n


def load_ytd_hr_pa(csv_path: Path) -> dict[int, tuple[int, int]]:
    """mlbam_id -> (HR, PA) from qualifying batters CSV (real season-to-date)."""
    out: dict[int, tuple[int, int]] = {}
    if not csv_path.is_file():
        return out
    with csv_path.open(newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            try:
                bid = int(str(row.get("mlbam_id", "")).strip())
                hr = int(float(row.get("HR") or 0))
                pa = int(float(row.get("PA") or 0))
            except (TypeError, ValueError):
                continue
            if bid > 0 and pa >= 0:
                out[bid] = (hr, pa)
    return out


def _beast_map(beast_rows: list[dict] | None) -> dict[tuple[int, int], float]:
    m: dict[tuple[int, int], float] = {}
    if not beast_rows:
        return m
    for r in beast_rows:
        try:
            b = int(r.get("batter_mlbam_id"))
            p = int(r.get("pitcher_mlbam_id"))
        except (TypeError, ValueError):
            continue
        m[(b, p)] = _f(r.get("adj_p_hr"))
    return m


def _driver_tags(r: dict) -> list[str]:
    tags: list[str] = []
    bmf = r.get("bvp_model_features") or {}
    try:
        if int(bmf.get("bvp_hr_count") or 0) > 0:
            tags.append("BvP_HR")
    except (TypeError, ValueError):
        pass
    if _f(r.get("p_bb")) > 0.10:
        tags.append("WalkRisk")
    if _f(r.get("career_hr")) > 0.045:
        tags.append("CareerPower")
    if not tags:
        tags.append("Matchup")
    return tags


def compute_residual_longshots(
    prod_rows: list[dict],
    *,
    beast_rows: list[dict] | None = None,
    qualifying_batters_csv: Path | None = None,
    config: ResidualConfig | None = None,
) -> list[dict]:
    """
    Return up to config.top_n rows, each dict:
      row, prior, lift, slate_pct, obviousness, ytd_hr, ytd_pa, ytd_rate,
      tags, beast_review, gem_score
    """
    if not prod_rows:
        return []
    c = config or ResidualConfig.defaults()

    root = Path(__file__).resolve().parents[1]
    csv_path = qualifying_batters_csv or (root / "data" / "raw" / "qualifying_batters_2026.csv")
    ytd_map = load_ytd_hr_pa(csv_path)
    beast_m = _beast_map(beast_rows)

    adj_list = sorted(_f(r.get("adj_p_hr")) for r in prod_rows)
    career_list = sorted(_f(r.get("career_hr")) for r in prod_rows)

    ytd_rates: list[float] = []
    for r in prod_rows:
        bid = r.get("batter_mlbam_id")
        try:
            bid_i = int(bid)
        except (TypeError, ValueError):
            continue
        t = ytd_map.get(bid_i)
        if not t:
            continue
        hr, pa = t
        if pa >= c.min_pa_for_ytd:
            ytd_rates.append(hr / pa)
    ytd_sorted = sorted(ytd_rates)

    scored: list[dict] = []
    for r in prod_rows:
        ap = _f(r.get("adj_p_hr"))
        if not (c.adj_hr_min <= ap <= c.adj_hr_max):
            continue
        ch = _f(r.get("career_hr"))
        prior = max(c.league_hr_prior, ch, c.prior_floor)
        lift = ap / max(prior, c.prior_floor)
        if lift < c.min_lift:
            continue
        slate_pct = _percentile_rank(adj_list, ap)
        if slate_pct > c.max_slate_pct:
            continue
        cf = _f(r.get("conf_hr"))
        if cf < c.min_conf_hr:
            continue

        career_pct = _percentile_rank(career_list, ch)

        ytd_hr = ytd_pa = None
        ytd_rate = None
        ytd_pct = 0.35
        try:
            bid_i = int(r.get("batter_mlbam_id"))
        except (TypeError, ValueError):
            bid_i = -1
        if bid_i > 0 and bid_i in ytd_map:
            yhr, ypa = ytd_map[bid_i]
            ytd_hr, ytd_pa = yhr, ypa
            if ypa >= c.min_pa_for_ytd:
                ytd_rate = yhr / ypa
                ytd_pct = _percentile_rank(ytd_sorted, ytd_rate) if ytd_sorted else 0.35

        if ytd_sorted and ytd_rate is not None:
            obs = c.w_slate * slate_pct + c.w_ytd * ytd_pct + c.w_career * career_pct
        else:
            wn = c.w_slate + c.w_career
            obs = (c.w_slate / wn) * slate_pct + (c.w_career / wn) * career_pct

        gem_score = lift * (1.0 - min(0.95, obs))

        try:
            b = int(r.get("batter_mlbam_id"))
            p = int(r.get("pitcher_mlbam_id"))
        except (TypeError, ValueError):
            b = p = -1
        b_adj = beast_m.get((b, p))
        beast_review = (
            b_adj is not None and abs(b_adj - ap) >= c.beast_hr_divergence
        )

        scored.append(
            {
                "row": r,
                "prior": prior,
                "lift": lift,
                "slate_pct": slate_pct,
                "obviousness": obs,
                "ytd_hr": ytd_hr,
                "ytd_pa": ytd_pa,
                "ytd_rate": ytd_rate,
                "tags": _driver_tags(r),
                "beast_review": beast_review,
                "gem_score": gem_score,
            }
        )

    scored.sort(key=lambda d: d["gem_score"], reverse=True)
    return scored[: c.top_n]


# Preloaded (P, batter HR-day map, (batter,pitcher) HR map, YYYY-MM-DD) for backtests.
SlateBacktestBundle = tuple[list[dict], dict, dict, str]


def load_slate_backtest_bundles(
    archive_root: Path,
    pa_path: Path,
    *,
    max_slates: int = 90,
) -> list[SlateBacktestBundle]:
    """Read Statcast and archived prediction JSONs once; reuse for many configs / grid search."""
    if not archive_root.is_dir() or not pa_path.is_file():
        return []
    try:
        import pandas as pd
    except Exception:
        return []

    pa = pd.read_parquet(pa_path, columns=["game_date", "batter", "pitcher", "events"])
    pa["game_date"] = pd.to_datetime(pa["game_date"], errors="coerce").dt.normalize()
    pa = pa.dropna(subset=["game_date", "batter", "pitcher"])
    pa["batter"] = pa["batter"].astype(int)
    pa["pitcher"] = pa["pitcher"].astype(int)
    pa["is_hr"] = (pa["events"].fillna("").str.lower() == "home_run").astype(int)
    pa = pa.drop(columns=["events"])

    slate_dirs: list[Path] = []
    for sub in sorted(archive_root.iterdir(), reverse=True):
        if not sub.is_dir():
            continue
        if not re.match(r"^\d{4}-\d{2}-\d{2}$", sub.name[:10]):
            continue
        if not (sub / "todays_matchup_predictions.json").is_file():
            continue
        slate_dirs.append(sub)
        if len(slate_dirs) >= max_slates:
            break

    out: list[SlateBacktestBundle] = []
    for sub in slate_dirs:
        sd_str = sub.name[:10]
        try:
            sd = pd.Timestamp(sd_str).normalize()
        except Exception:
            continue
        try:
            P = json.loads((sub / "todays_matchup_predictions.json").read_text(encoding="utf-8"))
        except Exception:
            continue
        if not isinstance(P, list) or not P:
            continue
        pa_d = pa.loc[pa["game_date"] == sd]
        if pa_d.empty:
            continue
        bb = pa_d.groupby("batter")["is_hr"].sum().to_dict()
        bbp = pa_d.groupby(["batter", "pitcher"])["is_hr"].sum().to_dict()
        out.append((P, bb, bbp, sd_str))
    return out


def evaluate_slate_bundles(
    bundles: list[SlateBacktestBundle],
    *,
    qualifying_batters_csv: Path | None = None,
    config: ResidualConfig | None = None,
) -> dict:
    """Same aggregation as ``evaluate_residual_history`` using preloaded bundles (fast for grids)."""
    c = config or ResidualConfig.defaults()
    if not bundles:
        return {}
    total_residual = 0
    total_band = 0
    residual_a = 0
    residual_b = 0
    band_a = 0
    band_b = 0
    slates_used = 0
    last_slate = ""

    for P, bb, bbp, sd_str in bundles:
        scored = compute_residual_longshots(
            P,
            qualifying_batters_csv=qualifying_batters_csv,
            config=c,
        )
        if not scored:
            continue

        slates_used += 1
        last_slate = sd_str

        for d in scored:
            r = d["row"]
            try:
                bid = int(r.get("batter_mlbam_id"))
                pid = int(r.get("pitcher_mlbam_id"))
            except (TypeError, ValueError):
                continue
            total_residual += 1
            if bb.get(bid, 0) > 0:
                residual_b += 1
            if bbp.get((bid, pid), 0) > 0:
                residual_a += 1

        for r in P:
            ap = _f(r.get("adj_p_hr"))
            if not (c.adj_hr_min <= ap <= c.adj_hr_max):
                continue
            try:
                bid = int(r.get("batter_mlbam_id"))
                pid = int(r.get("pitcher_mlbam_id"))
            except (TypeError, ValueError):
                continue
            total_band += 1
            if bb.get(bid, 0) > 0:
                band_b += 1
            if bbp.get((bid, pid), 0) > 0:
                band_a += 1

    if total_residual == 0:
        return {}

    return {
        "slates": slates_used,
        "last_slate": last_slate,
        "residual_rows": total_residual,
        "residual_tier_a_hits": residual_a,
        "residual_tier_b_hits": residual_b,
        "residual_tier_a_rate": residual_a / total_residual,
        "residual_tier_b_rate": residual_b / total_residual,
        "band_rows": total_band,
        "band_tier_a_hits": band_a,
        "band_tier_b_hits": band_b,
        "band_tier_a_rate": band_a / max(total_band, 1),
        "band_tier_b_rate": band_b / max(total_band, 1),
    }


def evaluate_residual_history(
    archive_root: Path,
    pa_path: Path,
    *,
    qualifying_batters_csv: Path | None = None,
    max_slates: int = 90,
    config: ResidualConfig | None = None,
) -> dict:
    """Backtest the residual screen over archived prediction snapshots.

    Returns aggregate counts and rates so the dashboard can render a
    short reliability blurb (Tier A vs listed starter; Tier B vs slate-day batter HRs).
    """
    bundles = load_slate_backtest_bundles(
        archive_root, pa_path, max_slates=max_slates
    )
    if not bundles:
        return {}
    return evaluate_slate_bundles(
        bundles,
        qualifying_batters_csv=qualifying_batters_csv,
        config=config,
    )
