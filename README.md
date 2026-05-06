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

## Daily run (typical)

From project root with `PYTHONPATH=src` and venv active:

```bash
PYTHONPATH=src python3 -u src/run_dual_model_daily.py --date YYYY-MM-DD
PYTHONPATH=src python3 -u src/gen_sections_11_13.py --recent-days 7
```

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
