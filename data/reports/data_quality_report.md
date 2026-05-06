# Data Quality Report

- Generated: 2026-05-05T22:05:18
- Predictions JSON: `data/reports/todays_matchup_predictions.json`
- Slate date: 2026-05-05
- Total rows: **390**

## Coverage by data_quality

| data_quality | rows | share |
|---|---:|---:|
| full_gbdt | 341 | 87.4% |
| x_is_none_league_avg | 49 | 12.6% |

## Imputation counters

| signal | rows | share |
|---|---:|---:|
| bpt_imputed (pitch-type EV missing) | 2 | 0.5% |
| g_roll_imputed (rolling NaN → league) | 0 | 0.0% |
| pitcher_profile_missing | 0 | 0.0% |
| pitcher_is_tbd | 0 | 0.0% |
| engagement_gate engaged | 363 | 93.1% |
| engagement_gate NOT engaged | 27 | 6.9% |

- Full-GBDT coverage: **87.4%** (warn threshold 85%).

## Hard-fail check (pure-default rows)

- HARD FAIL: 7 row(s) have no career, no YTD, AND no BvP evidence.

| Batter | Team | Pitcher | p_hit | p_hr | p_xbh |
|---|---|---|---:|---:|---:|
| Justin Foscue | TEX | Elmer Rodríguez | 0.2220 | 0.0310 | 0.0790 |
| Blake Dunn | CIN | Jameson Taillon | 0.2220 | 0.0310 | 0.0790 |
| César Salazar | HOU | Shohei Ohtani | 0.2220 | 0.0310 | 0.0790 |
| Zach Dezenzo | HOU | Shohei Ohtani | 0.2220 | 0.0310 | 0.0790 |
| Jhonny Pereda | SEA | Bryce Elder | 0.2220 | 0.0001 | 0.0790 |
| Sung-Mun Song | SD | Logan Webb | 0.2220 | 0.0310 | 0.0790 |
| César Prieto | STL | Brandon Sproat | 0.2220 | 0.0310 | 0.0790 |

