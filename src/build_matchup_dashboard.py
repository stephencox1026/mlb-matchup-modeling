"""
Build interactive HTML matchup dashboard for today's scheduled games.

Reads narrative engine output and produces a self-contained HTML file
with game cards, matchup predictions, and narrative insights.

Output: docs/matchup_dashboard.html
"""
import argparse
import json
import shutil
import sys, os
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, os.path.dirname(__file__))

from config import (
    BEAST_MODEL_DIR,
    BEAST_VAL_FEATURES,
    DOCS_DIR,
    EXPERIMENT_MODEL_DIR,
    EXPERIMENT_VAL_FEATURES,
    MASTER_DIR,
    RAW_DIR,
    REPORTS_DIR,
)
from matchup_tracking import (
    append_dual_model_predictions,
    append_matchup_tracking_rows,
    materialize_high_conf_tracking,
)
from narrative_engine import predict_matchups, pct, write_matchup_predictions_semantics_artifact
from starter_run_expectancy import predict_starter_runs, load_qualifying_pa_map
from render_section9 import render_section9_markdown


TIER_COLORS = {
    "Elite": "#22c55e",
    "Strong": "#3b82f6",
    "Average": "#a3a3a3",
    "Below Average": "#f59e0b",
    "Weak": "#ef4444",
}


def build_matchup_card(m):
    tier_color = TIER_COLORS.get(m["tier"], "#a3a3a3")
    bvp_html = f'<div class="bvp">{m["bvp_text"]}</div>' if m["bvp_text"] else ""
    reasons_html = ""
    if m.get("top_reasons"):
        reasons_html = '<div class="reasons">' + " | ".join(m["top_reasons"][:2]) + '</div>'

    return f"""
    <div class="matchup-card" data-hit="{m['p_hit']:.6f}" data-hr="{m['p_hr']:.6f}" data-k="{m['p_k']:.6f}"
         data-bb="{m['p_bb']:.6f}" data-xbh="{m['p_xbh']:.6f}" data-multi="{m['p_multi_hit']:.6f}">
      <div class="card-header">
        <div class="batter-info">
          <span class="batter-name">{m['batter_name']}</span>
          <span class="team-badge">{m['batter_team']}</span>
        </div>
        <span class="vs-label">vs</span>
        <div class="pitcher-info">
          <span class="pitcher-name">{m['pitcher_name']}</span>
          <span class="team-badge">{m['pitcher_team']}</span>
          <span class="hand-badge">{'L' if m['pitcher_throws'] == 'L' else 'R'}HP</span>
        </div>
        <span class="tier-badge" style="background:{tier_color}">{m['tier']}</span>
      </div>
      <div class="stats-grid">
        <div class="stat-item">
          <div class="stat-label">P(Hit)</div>
          <div class="stat-bar-container">
            <div class="stat-bar" style="width:{min(m['p_hit']*300, 100)}%;background:#22c55e"></div>
          </div>
          <div class="stat-value">{pct(m['p_hit'])}</div>
          <div class="stat-compare">Career: {pct(m['career_hit'])} | Lg: {pct(0.222)}</div>
        </div>
        <div class="stat-item">
          <div class="stat-label">P(Multi-Hit)</div>
          <div class="stat-bar-container">
            <div class="stat-bar" style="width:{min(m['p_multi_hit']*300, 100)}%;background:#3b82f6"></div>
          </div>
          <div class="stat-value">{pct(m['p_multi_hit'])}</div>
          <div class="stat-compare">3 PA vs starter</div>
        </div>
        <div class="stat-item">
          <div class="stat-label">P(HR)</div>
          <div class="stat-bar-container">
            <div class="stat-bar" style="width:{min(m['p_hr']*1000, 100)}%;background:#f59e0b"></div>
          </div>
          <div class="stat-value">{pct(m['p_hr'])}</div>
          <div class="stat-compare">Career: {pct(m['career_hr'])} | Lg: {pct(0.031)}</div>
        </div>
        <div class="stat-item">
          <div class="stat-label">P(K)</div>
          <div class="stat-bar-container">
            <div class="stat-bar" style="width:{min(m['p_k']*300, 100)}%;background:#ef4444"></div>
          </div>
          <div class="stat-value">{pct(m['p_k'])}</div>
          <div class="stat-compare">Career: {pct(m['career_k'])} | Lg: {pct(0.222)}</div>
        </div>
        <div class="stat-item">
          <div class="stat-label">P(BB)</div>
          <div class="stat-bar-container">
            <div class="stat-bar" style="width:{min(m['p_bb']*500, 100)}%;background:#8b5cf6"></div>
          </div>
          <div class="stat-value">{pct(m['p_bb'])}</div>
          <div class="stat-compare">Career: {pct(m['career_bb'])} | Lg: {pct(0.084)}</div>
        </div>
        <div class="stat-item">
          <div class="stat-label">P(XBH)</div>
          <div class="stat-bar-container">
            <div class="stat-bar" style="width:{min(m['p_xbh']*500, 100)}%;background:#06b6d4"></div>
          </div>
          <div class="stat-value">{pct(m['p_xbh'])}</div>
          <div class="stat-compare">Lg: {pct(0.079)}</div>
        </div>
      </div>
      <div class="narrative">
        <p><strong>Hit:</strong> {m['hit_narrative']}</p>
        <p><strong>Strikeout:</strong> {m['k_narrative']}</p>
        {bvp_html}
        {reasons_html}
      </div>
    </div>
    """


def build_html(results):
    game_sections = ""
    all_matchups_json = []

    for game_result in results:
        g = game_result["game"]
        matchups = sorted(game_result["matchups"], key=lambda x: x["p_hit"], reverse=True)

        cards_html = ""
        for m in matchups:
            cards_html += build_matchup_card(m)
            all_matchups_json.append({
                "batter": m["batter_name"], "batter_team": m["batter_team"],
                "pitcher": m["pitcher_name"], "pitcher_team": m["pitcher_team"],
                "tier": m["tier"],
                "p_hit": round(m["p_hit"], 4), "p_hr": round(m["p_hr"], 4),
                "p_k": round(m["p_k"], 4), "p_bb": round(m["p_bb"], 4),
                "p_xbh": round(m["p_xbh"], 4), "p_multi_hit": round(m["p_multi_hit"], 4),
            })

        game_sections += f"""
        <div class="game-section">
          <div class="game-header" onclick="this.parentElement.classList.toggle('collapsed')">
            <h2>{g['away_team']} @ {g['home_team']}</h2>
            <div class="pitchers">{g['away_pitcher_name']} vs {g['home_pitcher_name']}</div>
            <span class="matchup-count">{len(matchups)} matchups</span>
            <span class="expand-icon">&#9660;</span>
          </div>
          <div class="game-matchups">
            {cards_html}
          </div>
        </div>
        """

    today = game_result["game"]["game_date"] if results else "N/A"

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>MLB Matchup Predictions — {today}</title>
<style>
* {{ margin: 0; padding: 0; box-sizing: border-box; }}
body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
        background: #0f172a; color: #e2e8f0; line-height: 1.5; }}
.container {{ max-width: 1200px; margin: 0 auto; padding: 20px; }}
h1 {{ font-size: 1.8rem; margin-bottom: 5px; color: #f8fafc; }}
.subtitle {{ color: #94a3b8; margin-bottom: 20px; font-size: 0.9rem; }}
.model-info {{ background: #1e293b; border-radius: 8px; padding: 12px 16px;
               margin-bottom: 20px; font-size: 0.8rem; color: #94a3b8; }}
.model-info strong {{ color: #e2e8f0; }}
.filters {{ display: flex; gap: 10px; margin-bottom: 20px; flex-wrap: wrap; }}
.filters input, .filters select {{ background: #1e293b; border: 1px solid #334155;
  color: #e2e8f0; padding: 8px 12px; border-radius: 6px; font-size: 0.85rem; }}
.filters input {{ flex: 1; min-width: 200px; }}
.game-section {{ background: #1e293b; border-radius: 12px; margin-bottom: 16px;
                 overflow: hidden; }}
.game-section.collapsed .game-matchups {{ display: none; }}
.game-header {{ display: flex; align-items: center; gap: 12px; padding: 16px 20px;
                cursor: pointer; border-bottom: 1px solid #334155; }}
.game-header:hover {{ background: #263145; }}
.game-header h2 {{ font-size: 1.1rem; flex: 1; }}
.pitchers {{ color: #94a3b8; font-size: 0.85rem; }}
.matchup-count {{ background: #334155; padding: 2px 10px; border-radius: 12px;
                  font-size: 0.75rem; }}
.expand-icon {{ color: #64748b; transition: transform 0.2s; }}
.collapsed .expand-icon {{ transform: rotate(-90deg); }}
.game-matchups {{ padding: 8px; }}
.matchup-card {{ background: #0f172a; border-radius: 8px; padding: 16px;
                 margin: 8px; border: 1px solid #1e293b; }}
.matchup-card:hover {{ border-color: #3b82f6; }}
.card-header {{ display: flex; align-items: center; gap: 8px; margin-bottom: 12px;
                flex-wrap: wrap; }}
.batter-name {{ font-weight: 700; font-size: 1rem; }}
.pitcher-name {{ font-weight: 600; font-size: 0.9rem; color: #94a3b8; }}
.vs-label {{ color: #475569; font-size: 0.75rem; }}
.team-badge {{ background: #334155; padding: 1px 8px; border-radius: 4px;
               font-size: 0.7rem; font-weight: 600; }}
.hand-badge {{ background: #1e40af; padding: 1px 6px; border-radius: 4px;
               font-size: 0.65rem; }}
.tier-badge {{ padding: 2px 10px; border-radius: 12px; font-size: 0.7rem;
               font-weight: 700; color: #000; margin-left: auto; }}
.stats-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
               gap: 8px; margin-bottom: 12px; }}
.stat-item {{ background: #1e293b; border-radius: 6px; padding: 8px 10px; }}
.stat-label {{ font-size: 0.7rem; color: #64748b; text-transform: uppercase;
               letter-spacing: 0.05em; }}
.stat-bar-container {{ height: 4px; background: #334155; border-radius: 2px;
                       margin: 4px 0; overflow: hidden; }}
.stat-bar {{ height: 100%; border-radius: 2px; transition: width 0.5s; }}
.stat-value {{ font-size: 1.1rem; font-weight: 700; }}
.stat-compare {{ font-size: 0.65rem; color: #64748b; }}
.narrative {{ font-size: 0.8rem; color: #94a3b8; border-top: 1px solid #1e293b;
              padding-top: 10px; }}
.narrative p {{ margin-bottom: 4px; }}
.narrative strong {{ color: #e2e8f0; }}
.bvp {{ background: #1a1a2e; border-left: 3px solid #3b82f6; padding: 6px 10px;
        margin-top: 6px; border-radius: 0 4px 4px 0; font-size: 0.8rem; }}
.reasons {{ color: #22c55e; margin-top: 4px; font-size: 0.75rem; }}
@media (max-width: 768px) {{
  .stats-grid {{ grid-template-columns: repeat(3, 1fr); }}
  .card-header {{ flex-direction: column; align-items: flex-start; }}
  .tier-badge {{ margin-left: 0; }}
}}
</style>
</head>
<body>
<div class="container">
  <h1>MLB Matchup Predictions</h1>
  <p class="subtitle">{today} &mdash; {len(results)} games, {sum(len(g['matchups']) for g in results)} batter-pitcher matchups</p>

  <div class="model-info">
    <strong>Model:</strong> Calibrated HistGradientBoosting on 1.96M PAs (2015-2024), validated on 297K PAs (2025).
    <strong>Features:</strong> 78 (batter rolling stats + pitcher arsenal/velo/spin/movement + BvP history + pitch-type matchups).
    <strong>AUC:</strong> Hit 0.78 | K 0.86 | BB 0.96 | HR 0.76.
    <strong>Sim:</strong> 10K Monte Carlo, 3 PAs vs starter.
  </div>

  <div class="filters">
    <input type="text" id="search" placeholder="Search batter or pitcher..." oninput="filterCards()">
    <select id="tierFilter" onchange="filterCards()">
      <option value="">All Tiers</option>
      <option value="Elite">Elite</option>
      <option value="Strong">Strong</option>
      <option value="Average">Average</option>
      <option value="Below Average">Below Average</option>
      <option value="Weak">Weak</option>
    </select>
    <select id="sortBy" onchange="sortCards()">
      <option value="p_hit">Sort by P(Hit)</option>
      <option value="p_hr">Sort by P(HR)</option>
      <option value="p_k">Sort by P(K) ↑</option>
      <option value="p_multi_hit">Sort by P(Multi-Hit)</option>
    </select>
  </div>

  {game_sections}
</div>
<script>
const allMatchups = {json.dumps(all_matchups_json)};

function filterCards() {{
  const q = document.getElementById('search').value.toLowerCase();
  const tier = document.getElementById('tierFilter').value;
  document.querySelectorAll('.matchup-card').forEach(card => {{
    const text = card.textContent.toLowerCase();
    const tierBadge = card.querySelector('.tier-badge')?.textContent || '';
    const matchSearch = !q || text.includes(q);
    const matchTier = !tier || tierBadge === tier;
    card.style.display = (matchSearch && matchTier) ? '' : 'none';
  }});
}}

function sortCards() {{
  const sortKey = document.getElementById('sortBy').value;
  const attr = sortKey === 'p_hit' ? 'hit' : sortKey === 'p_hr' ? 'hr' : sortKey === 'p_k' ? 'k'
    : sortKey === 'p_bb' ? 'bb' : sortKey === 'p_xbh' ? 'xbh' : sortKey === 'p_multi_hit' ? 'multi' : 'hit';
  document.querySelectorAll('.game-matchups').forEach(container => {{
    const cards = Array.from(container.children);
    cards.sort((a, b) => {{
      const aVal = parseFloat(a.dataset[attr]) || 0;
      const bVal = parseFloat(b.dataset[attr]) || 0;
      return sortKey === 'p_k' ? aVal - bVal : bVal - aVal;
    }});
    cards.forEach(c => container.appendChild(c));
  }});
}}
</script>
</body>
</html>"""
    return html


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dual-model",
        action="store_true",
        help="Also run experiment bundle and append data/tracking/matchup_dual_model_predictions.parquet",
    )
    parser.add_argument(
        "--skip-dq",
        action="store_true",
        help="Skip the runtime data-quality pre-check (break-glass only)",
    )
    args = parser.parse_args()

    if not args.skip_dq:
        from data_quality_check import run_pre_checks
        ok, _results = run_pre_checks(verbose=True)
        if not ok:
            print("\nData-quality pre-check FAILED. Refresh source data or pass --skip-dq.")
            sys.exit(2)

    print("Building matchup dashboard...")
    results = predict_matchups(model_source="prod")

    html = build_html(results)

    out = DOCS_DIR / "matchup_dashboard.html"
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        f.write(html)

    total = sum(len(g["matchups"]) for g in results)
    print(f"\nDashboard built: {len(results)} games, {total} matchups")
    print(f"Saved → {out}")

    results_out = REPORTS_DIR / "todays_matchup_predictions.json"
    flat = []
    for gr in results:
        for m in gr["matchups"]:
            flat.append(m)
    with open(results_out, "w") as f:
        json.dump(flat, f, indent=2, default=str)
    write_matchup_predictions_semantics_artifact(REPORTS_DIR)
    print(f"Predictions JSON → {results_out}")

    if not args.skip_dq:
        from data_quality_check import run_post_checks
        ok_post, _summary = run_post_checks(results_out, verbose=True)
        if not ok_post:
            print(
                "\nData-quality POST check found pure-default rows. Inspect "
                "data/reports/data_quality_report.md before trusting the slate."
            )

    beast_ok = (
        BEAST_MODEL_DIR.is_dir()
        and (BEAST_MODEL_DIR / "feature_columns.json").exists()
        and BEAST_VAL_FEATURES.is_file()
    )
    if beast_ok:
        print("Running Beast bundle (Matchups · Beast tab)...")
        beast_results = predict_matchups(
            model_dir=BEAST_MODEL_DIR,
            val_features_path=BEAST_VAL_FEATURES,
            model_source="beast",
        )
        beast_flat = [m for gr in beast_results for m in gr["matchups"]]
        beast_out = REPORTS_DIR / "todays_matchup_predictions_beast.json"
        with open(beast_out, "w") as f:
            json.dump(beast_flat, f, indent=2, default=str)
        print(f"Beast predictions JSON → {beast_out} ({len(beast_flat)} matchups)")
    else:
        print(
            "WARN: Skipped beast predictions — install bundle at",
            BEAST_MODEL_DIR,
            "and",
            BEAST_VAL_FEATURES,
        )

    run_ts = datetime.now(timezone.utc).isoformat()
    n_trk = append_matchup_tracking_rows(flat, run_ts)
    materialize_high_conf_tracking()
    print(f"Tracking Parquet: {n_trk} rows appended → data/tracking/matchup_predictions_runs.parquet")
    print("High-confidence slice → data/tracking/matchup_predictions_runs_high_conf.parquet")

    if args.dual_model:
        if not EXPERIMENT_MODEL_DIR.is_dir() or not (EXPERIMENT_MODEL_DIR / "feature_columns.json").exists():
            print("WARN: --dual-model skipped: experiment models missing at", EXPERIMENT_MODEL_DIR)
        elif not EXPERIMENT_VAL_FEATURES.exists():
            print("WARN: --dual-model skipped: missing", EXPERIMENT_VAL_FEATURES)
        else:
            print("Running experiment bundle for dual-model diff...")
            exp_results = predict_matchups(
                model_dir=EXPERIMENT_MODEL_DIR,
                val_features_path=EXPERIMENT_VAL_FEATURES,
                model_source="exp",
            )
            exp_flat = [m for gr in exp_results for m in gr["matchups"]]
            n_dual = append_dual_model_predictions(
                flat,
                exp_flat,
                run_ts,
                prod_model_dir=str(MASTER_DIR / "models"),
                exp_model_dir=str(EXPERIMENT_MODEL_DIR),
                prod_val_path=str(MASTER_DIR / "features_val_league.parquet"),
                exp_val_path=str(EXPERIMENT_VAL_FEATURES),
            )
            print(f"Dual-model Parquet: {n_dual} rows → data/tracking/matchup_dual_model_predictions.parquet")
            try:
                from compare_dual_model_predictions import write_dual_summary

                out = write_dual_summary(run_timestamp=run_ts)
                print(f"Dual-model summary → {out}")
            except Exception as ex:
                print("WARN: dual-model summary failed:", ex)

    # Snapshot for top-10 prediction audit (same JSON; does not affect model).
    if results:
        slate = str(results[0]["game"].get("game_date", "")).strip()
        if slate:
            arch_dir = Path(REPORTS_DIR) / "archive" / slate
            arch_dir.mkdir(parents=True, exist_ok=True)
            arch_pred = arch_dir / "todays_matchup_predictions.json"
            shutil.copy2(results_out, arch_pred)
            print(f"Archived predictions → {arch_pred}")
            beast_root = REPORTS_DIR / "todays_matchup_predictions_beast.json"
            if beast_root.is_file():
                arch_beast = arch_dir / "todays_matchup_predictions_beast.json"
                shutil.copy2(beast_root, arch_beast)
                print(f"Archived beast predictions → {arch_beast}")

    pa_map = load_qualifying_pa_map()
    starter_rows = predict_starter_runs(results, pa_map)
    starter_out = REPORTS_DIR / "todays_starter_run_expectancies.json"
    with open(starter_out, "w") as f:
        json.dump(starter_rows, f, indent=2)
    print(f"Starter run expectancies → {starter_out}")
    section9_out = REPORTS_DIR / "section_9.md"
    with open(section9_out, "w") as f:
        f.write(render_section9_markdown(starter_rows))
    print(f"Section 9 markdown → {section9_out}")


if __name__ == "__main__":
    main()
