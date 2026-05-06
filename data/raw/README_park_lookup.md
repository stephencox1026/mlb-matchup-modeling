# `park_lookup.csv` — ballpark exposure + retractable-roof priors

## Columns (see header row in CSV)

- **`home_team`**: 3-letter MLB home-team abbrev (normalized; matches `todays_matchups.json` / Statcast `home_team` after [park_metrics.normalize_mlb_home_team](src/park_metrics.py)).
- **`enclosure`**
  - `outdoor` — always **outside** the fixed roof: open-air stadia (naturally ventilated, no operable roof).
  - `fixed_dome` — always **inside** a fixed roof: play is to an enclosed shell (e.g. Tropicana Field).
  - `retractable` — can be **either** open to the outside **or** closed under a roof; we model *daily* **P(roof open)** and **P(roof closed) = 1 − P(open)** with **monthly priors** (`p_open_01`…`p_open_12`). These are **seasonal priors** (rough climate / usage), not a live forecast feed.

- **`venue_id`**, **`venue_name`**: reference only; join key for Statcast/Stats API is often `home_team`.
- **`park_hook_reserved`**: reserved column (always `0` in this file); keeps the schema stable for downstream tooling.
- **`notes`**: free text; documents assumptions.

## `park_*` model features (engineered in code)

- **Type one-hots (mutually exclusive):** `park_stadium_type_outdoor`, `park_stadium_type_fixed_dome`, `park_stadium_type_retractable`
- **Roof (only meaningful for retractable; fixed by construction for other types):** `park_p_roof_open`, `park_p_roof_closed`
- **Effective “plays outside / inside (enclosed)”:** `park_p_effective_plays_outside`, `park_p_effective_plays_enclosed` — sum to 1; interpret as **expected** exposure for that *calendar month* for retractable sites.

## Updating priors

Edit monthly `p_open_*` for `retractable` rows, or re-run a future calibration from historical roof-open labels if you have them. Priors v1 are hand-smoothed heuristics (colder months lean closed for northern domes, etc.).
