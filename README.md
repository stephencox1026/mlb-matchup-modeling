> **Production daily ML pipeline** with automated data-quality gates and model calibration — an MLOps demonstration of models that run reliably in production.

# LAD vs LHP Analysis

Daily matchup modeling pipeline: Statcast-backed features, multi-target GBDT models (HR / hit / XBH / etc.), calibration, dashboards, and Beast variant.

## Clone and environment

```bash
git clone <YOUR_GITHUB_URL>
cd "LAD vs LHP Analysis"
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Large `.parquet` training caches and `.pkl` model weights are **not** in this repo (see `.gitignore`). Refresh data and retrain or copy artifacts from your machine backup if you need a full runnable snapshot.

## Daily run (recommended)

**`--date`** is always the **prediction slate** (the calendar day of the games you are scoring), not necessarily “today.”

### One command (lock + optional prior-day outcomes)

[`scripts/daily_run.sh`](scripts/daily_run.sh) uses `flock` so two runs cannot overlap. Optionally set **`D0`** to the **completed** calendar day first: append that day’s Statcast into the league parquet, then fill tracking outcomes (order matters).

```bash
# Prediction slate only (same as run_dual_model_daily)
./scripts/daily_run.sh 2026-05-10

# After games: refresh Statcast for D0, fill outcomes for D0, then run the slate for D1
D0=2026-05-09 ./scripts/daily_run.sh 2026-05-10
```

### Spine only (what `run_dual_model_daily.py` does)

From project root with `PYTHONPATH=src` and venv active:

```bash
PYTHONPATH=src python3 -u src/run_dual_model_daily.py --date YYYY-MM-DD
PYTHONPATH=src python3 -u src/gen_sections_11_13.py --recent-days 7
```

That script: refreshes [`data/raw/todays_matchups.json`](data/raw/todays_matchups.json), runs **data-quality pre/post checks**, appends **prior calendar day** Statcast (unless `--skip-statcast-append`), runs **prod (Beast) + exp + legacy + recent365 + Beast tab JSON**, writes reports, **`gen_matchup_dashboard_html`**, **`gen_beast_html`** → [`docs/beast.html`](docs/beast.html) (+ mirror under `data/reports/beast.html`), archives to [`data/reports/archive/<date>/`](data/reports/archive/), runs **`verify_matchup_dashboard_outputs`** (fails the process on mismatch), and appends a line to **`data/reports/daily_run_manifest.jsonl`**.

**Roster hard gate:** before tracking append, predictions are checked against the slate roster in `todays_matchups.json` ([`src/verify_roster_consistency.py`](src/verify_roster_consistency.py)) so batters/teams/game_pk cannot drift silently.

**Atomic JSON:** main prediction JSON files are written via temp+replace ([`src/atomic_io.py`](src/atomic_io.py)). A backup of the previous root prod file is saved as `todays_matchup_predictions.json.prev` when present.

See `.cursor/rules/` and `docs/` for output conventions and model notes.

## Referencing this repo on GitHub

- **Stable pointer:** use a **tag** (e.g. `v0.1.0`) or the **full commit SHA** in links:
  - `https://github.com/<user>/<repo>/tree/<commit-sha>`
- **Issues / PRs:** reference `owner/repo#123` or paste the commit URL.
- After your first push, create a release from tag `v0.1.0` if you want a downloadable snapshot label.

## Publishing to GitHub (first time)

1. Create an empty repository on GitHub (no README/license if you already have them locally), e.g. `lad-vs-lhp-analysis`.
2. From this directory:

```bash
git remote add origin https://github.com/<YOUR_USER>/<YOUR_REPO>.git
git branch -M main
git push -u origin main
git push origin v0.1.0   # optional: after tagging (see below)
```

3. Optional version tag (already created locally on first import commit):

```bash
git tag -a v0.1.0 -m "Initial GitHub import"
git push origin v0.1.0
```
