"""Beta-Binomial shrinkage predictions for each Dodger vs LHP.

Blends 2026 YTD counts with career same-split priors to produce
'expected true-talent' rates and per-game probabilities.
NOT a trained ML model — honest Bayesian smoothing only.
"""
import numpy as np
import pandas as pd
from scipy import stats as sp_stats
from config import MASTER_DIR, REPORTS_DIR

EXPECTED_PA_PER_GAME = 4.0


def load_data():
    master = pd.read_parquet(MASTER_DIR / "lad_hitters_sprint.parquet")
    statcast = pd.read_csv(REPORTS_DIR / "statcast_contact_summary.csv")
    return master, statcast


def beta_shrink(career_successes, career_trials, ytd_successes, ytd_trials,
                prior_weight=0.6):
    """Bayesian Beta-Binomial posterior mean.

    prior_weight controls how much career dominates when YTD sample is small.
    As ytd_trials grows, the posterior shifts toward the YTD rate.
    """
    if career_trials == 0:
        if ytd_trials == 0:
            return 0.0
        return ytd_successes / ytd_trials

    career_rate = career_successes / career_trials
    alpha_prior = career_rate * career_trials * prior_weight
    beta_prior = (1 - career_rate) * career_trials * prior_weight

    alpha_prior = max(alpha_prior, 1.0)
    beta_prior = max(beta_prior, 1.0)

    alpha_post = alpha_prior + ytd_successes
    beta_post = beta_prior + (ytd_trials - ytd_successes)

    return alpha_post / (alpha_post + beta_post)


def credible_interval(career_successes, career_trials, ytd_successes, ytd_trials,
                      prior_weight=0.6, ci=0.80):
    if career_trials == 0:
        return (0.0, 1.0)
    career_rate = career_successes / career_trials
    alpha_prior = max(career_rate * career_trials * prior_weight, 1.0)
    beta_prior = max((1 - career_rate) * career_trials * prior_weight, 1.0)

    alpha_post = alpha_prior + ytd_successes
    beta_post = beta_prior + (ytd_trials - ytd_successes)

    low = sp_stats.beta.ppf((1 - ci) / 2, alpha_post, beta_post)
    high = sp_stats.beta.ppf(1 - (1 - ci) / 2, alpha_post, beta_post)
    return (round(low, 3), round(high, 3))


def build_predictions(master: pd.DataFrame, statcast: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for name in master["name"].unique():
        ytd_lhp = master[(master["name"] == name) & (master["scope"] == "2026") & (master["split"] == "vs LHP")]
        car_lhp = master[(master["name"] == name) & (master["scope"] == "Career") & (master["split"] == "vs LHP")]
        ytd_all = master[(master["name"] == name) & (master["scope"] == "2026") & (master["split"] == "Overall")]

        if ytd_lhp.empty or car_lhp.empty:
            continue

        y = ytd_lhp.iloc[0]
        c = car_lhp.iloc[0]
        o = ytd_all.iloc[0] if not ytd_all.empty else y

        pa_ytd = int(y["PA"])
        ab_ytd = int(y["AB"])
        pa_car = int(c["PA"])
        ab_car = int(c["AB"])

        h_rate = beta_shrink(c["H"], ab_car, y["H"], ab_ytd)
        hr_rate = beta_shrink(c["HR"], pa_car, y["HR"], pa_ytd)
        k_rate = beta_shrink(c["SO"], pa_car, y["SO"], pa_ytd)
        bb_rate = beta_shrink(c["BB"], pa_car, y["BB"], pa_ytd)
        iso_ytd = float(y["SLG"]) - float(y["BA"])
        iso_car = float(c["SLG"]) - float(c["BA"])
        iso_shrunk = 0.6 * iso_car + 0.4 * iso_ytd if pa_ytd >= 5 else iso_car

        h_ci = credible_interval(c["H"], ab_car, y["H"], ab_ytd)
        hr_ci = credible_interval(c["HR"], pa_car, y["HR"], pa_ytd)

        p_hit_game = 1 - (1 - h_rate) ** EXPECTED_PA_PER_GAME
        p_hr_game = 1 - (1 - hr_rate) ** EXPECTED_PA_PER_GAME
        p_multi_hit = 1 - ((1 - h_rate) ** EXPECTED_PA_PER_GAME +
                           EXPECTED_PA_PER_GAME * h_rate * (1 - h_rate) ** (EXPECTED_PA_PER_GAME - 1))
        exp_tb_pa = h_rate * (1 + iso_shrunk / max(h_rate, 0.001)) if h_rate > 0 else 0
        exp_tb_game = exp_tb_pa * EXPECTED_PA_PER_GAME

        sc_row = statcast[statcast["name"] == name]
        avg_ev = float(sc_row["avg_EV"].iloc[0]) if not sc_row.empty else None
        barrel_pct = float(sc_row["barrel_pct"].iloc[0]) if not sc_row.empty else None
        hard_hit = float(sc_row["hard_hit_pct"].iloc[0]) if not sc_row.empty else None

        reliability = "HIGH" if pa_ytd >= 25 else ("MED" if pa_ytd >= 10 else "LOW")

        rows.append({
            "name": name,
            "PA_2026_vsLHP": pa_ytd,
            "PA_career_vsLHP": pa_car,
            "BA_2026": float(y["BA"]),
            "BA_career": float(c["BA"]),
            "BA_shrunk": round(h_rate, 3),
            "BA_80ci_lo": h_ci[0],
            "BA_80ci_hi": h_ci[1],
            "OPS_2026": float(y["OPS"]),
            "OPS_career": float(c["OPS"]),
            "HR_per_PA_shrunk": round(hr_rate, 4),
            "HR_80ci_lo": hr_ci[0],
            "HR_80ci_hi": hr_ci[1],
            "K_pct_shrunk": round(k_rate * 100, 1),
            "BB_pct_shrunk": round(bb_rate * 100, 1),
            "ISO_shrunk": round(iso_shrunk, 3),
            "P_hit_game": round(p_hit_game, 3),
            "P_HR_game": round(p_hr_game, 3),
            "P_multi_hit_game": round(p_multi_hit, 3),
            "exp_TB_game": round(exp_tb_game, 2),
            "avg_EV": avg_ev,
            "barrel_pct": barrel_pct,
            "hard_hit_pct": hard_hit,
            "reliability": reliability,
        })

    df = pd.DataFrame(rows).sort_values("P_hit_game", ascending=False).reset_index(drop=True)
    return df


if __name__ == "__main__":
    master, statcast = load_data()
    preds = build_predictions(master, statcast)
    preds.to_csv(REPORTS_DIR / "predictions_vs_lhp.csv", index=False)
    print(preds[["name", "BA_shrunk", "HR_per_PA_shrunk", "P_hit_game", "P_HR_game",
                  "P_multi_hit_game", "exp_TB_game", "reliability"]].to_string(index=False))
