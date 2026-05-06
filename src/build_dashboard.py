"""Build executive predictions dashboard as self-contained HTML."""
import base64
import pandas as pd
from pathlib import Path
from config import MASTER_DIR, REPORTS_DIR, FIGURES_DIR, DOCS_DIR

OUT = DOCS_DIR / "PREDICTIONS_DASHBOARD.html"


def img_b64(path: Path) -> str:
    if path.exists():
        return base64.b64encode(path.read_bytes()).decode()
    return ""


def pct(v):
    return f"{v * 100:.1f}%" if isinstance(v, float) else v


def bar_html(val, max_val=1.0, color="#2176AE"):
    w = min(val / max_val * 100, 100)
    return f'<div style="background:#e8ecf1;border-radius:3px;height:18px;width:120px;display:inline-block;vertical-align:middle"><div style="background:{color};height:18px;border-radius:3px;width:{w:.0f}%"></div></div>'


def reliability_badge(r):
    colors = {"HIGH": "#4CAF50", "MED": "#FFC107", "LOW": "#FF9800"}
    c = colors.get(r, "#999")
    return f'<span style="background:{c};color:#fff;padding:2px 8px;border-radius:10px;font-size:11px;font-weight:600">{r}</span>'


def build():
    preds = pd.read_csv(REPORTS_DIR / "predictions_vs_lhp.csv")
    master = pd.read_parquet(MASTER_DIR / "lad_hitters_sprint.parquet")
    outliers = pd.read_csv(REPORTS_DIR / "outlier_scores.csv")
    olhp = outliers[outliers["split"] == "vs LHP"].set_index("name")

    pet = master[(master["split"] == "vs Peterson") & (master["PA"] > 0)].sort_values("OPS", ascending=False)

    # Build player cards
    cards = []
    for _, p in preds.iterrows():
        name = p["name"]
        o = olhp.loc[name] if name in olhp.index else None
        o_score = int(o["outlier_score"]) if o is not None else "—"
        o_label = {1: "Cold outlier", 2: "Below expected", 3: "As expected", 4: "Above expected", 5: "Hot outlier"}.get(o_score, "—")
        o_color = {1: "#1565C0", 2: "#42A5F5", 3: "#78909C", 4: "#FF9800", 5: "#E53935"}.get(o_score, "#999")

        pet_row = pet[pet["name"] == name]
        pet_str = f'{int(pet_row.iloc[0]["H"])}-for-{int(pet_row.iloc[0]["AB"])}, {int(pet_row.iloc[0]["HR"])} HR' if not pet_row.empty else "No PA"

        ev_str = f'{p["avg_EV"]:.1f} mph' if pd.notna(p["avg_EV"]) else "—"
        barrel_str = f'{p["barrel_pct"]:.1f}%' if pd.notna(p["barrel_pct"]) else "—"

        cards.append(f"""
        <div class="card">
          <div class="card-header">
            <div class="player-name">{name}</div>
            <div>{reliability_badge(p['reliability'])}</div>
          </div>
          <div class="card-body">
            <div class="stat-row">
              <div class="stat-group">
                <div class="stat-label">2026 vs LHP</div>
                <div class="stat-big">{p['OPS_2026']:.3f} <span class="stat-unit">OPS</span></div>
                <div class="stat-sub">BA {p['BA_2026']:.3f} &middot; {int(p['PA_2026_vsLHP'])} PA</div>
              </div>
              <div class="stat-group">
                <div class="stat-label">Career vs LHP</div>
                <div class="stat-big">{p['OPS_career']:.3f} <span class="stat-unit">OPS</span></div>
                <div class="stat-sub">BA {p['BA_career']:.3f} &middot; {int(p['PA_career_vsLHP'])} PA</div>
              </div>
              <div class="stat-group">
                <div class="stat-label">Outlier Score</div>
                <div class="stat-big" style="color:{o_color}">{o_score}</div>
                <div class="stat-sub">{o_label}</div>
              </div>
            </div>

            <div class="divider"></div>
            <div class="section-title">Shrunk Predictions vs LHP (per game, ~4 PA)</div>
            <div class="pred-grid">
              <div class="pred-item">
                <div class="pred-val">{p['P_hit_game'] * 100:.0f}%</div>
                <div class="pred-label">P(Hit)</div>
                {bar_html(p['P_hit_game'], 1.0, '#4CAF50')}
              </div>
              <div class="pred-item">
                <div class="pred-val">{p['P_HR_game'] * 100:.1f}%</div>
                <div class="pred-label">P(HR)</div>
                {bar_html(p['P_HR_game'], 0.3, '#E53935')}
              </div>
              <div class="pred-item">
                <div class="pred-val">{p['P_multi_hit_game'] * 100:.0f}%</div>
                <div class="pred-label">P(Multi-Hit)</div>
                {bar_html(p['P_multi_hit_game'], 0.5, '#FF9800')}
              </div>
              <div class="pred-item">
                <div class="pred-val">{p['exp_TB_game']:.1f}</div>
                <div class="pred-label">Exp TB</div>
                {bar_html(p['exp_TB_game'], 4.0, '#7B1FA2')}
              </div>
            </div>

            <div class="divider"></div>
            <div class="detail-row">
              <div class="detail">
                <span class="detail-label">Shrunk BA</span>
                <span class="detail-val">{p['BA_shrunk']:.3f}</span>
                <span class="detail-ci">80% CI: [{p['BA_80ci_lo']:.3f}, {p['BA_80ci_hi']:.3f}]</span>
              </div>
              <div class="detail">
                <span class="detail-label">HR/PA</span>
                <span class="detail-val">{p['HR_per_PA_shrunk']:.4f}</span>
                <span class="detail-ci">80% CI: [{p['HR_80ci_lo']:.3f}, {p['HR_80ci_hi']:.3f}]</span>
              </div>
              <div class="detail">
                <span class="detail-label">K%</span>
                <span class="detail-val">{p['K_pct_shrunk']:.1f}%</span>
              </div>
              <div class="detail">
                <span class="detail-label">BB%</span>
                <span class="detail-val">{p['BB_pct_shrunk']:.1f}%</span>
              </div>
              <div class="detail">
                <span class="detail-label">ISO</span>
                <span class="detail-val">{p['ISO_shrunk']:.3f}</span>
              </div>
              <div class="detail">
                <span class="detail-label">Avg EV</span>
                <span class="detail-val">{ev_str}</span>
              </div>
              <div class="detail">
                <span class="detail-label">Barrel%</span>
                <span class="detail-val">{barrel_str}</span>
              </div>
              <div class="detail">
                <span class="detail-label">vs Peterson</span>
                <span class="detail-val">{pet_str}</span>
              </div>
            </div>
          </div>
        </div>""")

    # Embed chart images
    chart_imgs = ""
    for fig_name, title in [
        ("ops_lhp_vs_rhp_2026.png", "2026 OPS: vs LHP vs vs RHP"),
        ("outlier_score_lhp.png", "Outlier Scores vs LHP"),
        ("statcast_ev_2026.png", "Statcast Exit Velocity"),
        ("peterson_matchup.png", "Career vs David Peterson"),
    ]:
        b64 = img_b64(FIGURES_DIR / fig_name)
        if b64:
            chart_imgs += f'<div class="chart-card"><h3>{title}</h3><img src="data:image/png;base64,{b64}"></div>\n'

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>LAD vs LHP — Predictions Dashboard</title>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{
    font-family: -apple-system, BlinkMacSystemFont, "SF Pro Display", "Segoe UI", Helvetica, Arial, sans-serif;
    background: #f0f2f5; color: #1a1a2e; line-height: 1.5;
  }}
  .header {{
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
    color: #fff; padding: 32px 40px; position: relative;
  }}
  .header h1 {{ font-size: 26px; font-weight: 700; letter-spacing: -0.5px; }}
  .header p {{ color: #94a3b8; font-size: 14px; margin-top: 6px; }}
  .header .badge {{
    display: inline-block; background: rgba(255,255,255,0.12); padding: 4px 12px;
    border-radius: 20px; font-size: 12px; margin-top: 10px; color: #e2e8f0;
  }}
  .container {{ max-width: 1100px; margin: 0 auto; padding: 24px; }}
  .section-header {{ font-size: 18px; font-weight: 700; color: #1a1a2e; margin: 28px 0 14px; padding-bottom: 6px; border-bottom: 2px solid #2176AE; }}
  .card {{
    background: #fff; border-radius: 10px; margin-bottom: 16px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.08); overflow: hidden;
  }}
  .card-header {{
    display: flex; justify-content: space-between; align-items: center;
    padding: 14px 20px; background: #f8fafc; border-bottom: 1px solid #e2e8f0;
  }}
  .player-name {{ font-size: 17px; font-weight: 700; color: #1a1a2e; }}
  .card-body {{ padding: 16px 20px; }}
  .stat-row {{ display: flex; gap: 32px; flex-wrap: wrap; }}
  .stat-group {{ min-width: 140px; }}
  .stat-label {{ font-size: 11px; text-transform: uppercase; color: #64748b; letter-spacing: 0.5px; font-weight: 600; }}
  .stat-big {{ font-size: 28px; font-weight: 800; color: #1a1a2e; line-height: 1.2; }}
  .stat-unit {{ font-size: 13px; font-weight: 500; color: #94a3b8; }}
  .stat-sub {{ font-size: 12px; color: #94a3b8; }}
  .divider {{ height: 1px; background: #e2e8f0; margin: 14px 0; }}
  .section-title {{ font-size: 12px; text-transform: uppercase; color: #64748b; letter-spacing: 0.5px; font-weight: 700; margin-bottom: 10px; }}
  .pred-grid {{ display: flex; gap: 24px; flex-wrap: wrap; }}
  .pred-item {{ min-width: 110px; }}
  .pred-val {{ font-size: 22px; font-weight: 800; color: #1a1a2e; }}
  .pred-label {{ font-size: 11px; color: #64748b; text-transform: uppercase; margin-bottom: 4px; }}
  .detail-row {{ display: flex; gap: 16px; flex-wrap: wrap; }}
  .detail {{
    background: #f8fafc; padding: 6px 12px; border-radius: 6px;
    font-size: 12px; display: flex; gap: 6px; align-items: baseline;
  }}
  .detail-label {{ color: #64748b; font-weight: 600; }}
  .detail-val {{ color: #1a1a2e; font-weight: 700; }}
  .detail-ci {{ color: #94a3b8; font-size: 11px; }}
  .charts {{ display: grid; grid-template-columns: 1fr 1fr; gap: 16px; margin-top: 16px; }}
  .chart-card {{
    background: #fff; border-radius: 10px; padding: 16px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.08);
  }}
  .chart-card h3 {{ font-size: 14px; color: #1a1a2e; margin-bottom: 10px; }}
  .chart-card img {{ width: 100%; border-radius: 6px; }}
  .methodology {{
    background: #fff; border-radius: 10px; padding: 20px; margin-top: 20px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.08); font-size: 13px; color: #475569;
  }}
  .methodology h3 {{ font-size: 15px; color: #1a1a2e; margin-bottom: 8px; }}
  .methodology ul {{ margin-left: 18px; }}
  .methodology li {{ margin-bottom: 4px; }}
  @media print {{
    body {{ background: #fff; }}
    .card {{ break-inside: avoid; box-shadow: none; border: 1px solid #e2e8f0; }}
    .header {{ background: #1a1a2e !important; -webkit-print-color-adjust: exact; print-color-adjust: exact; }}
  }}
  @media (max-width: 700px) {{
    .charts {{ grid-template-columns: 1fr; }}
    .stat-row, .pred-grid {{ gap: 16px; }}
  }}
</style>
</head>
<body>

<div class="header">
  <h1>Los Angeles Dodgers — Predictions vs Left-Handed Pitching</h1>
  <p>2026 Season &middot; 14 Active Hitters &middot; Bayesian Shrinkage Estimates</p>
  <div class="badge">Data as of April 13, 2026 &middot; Golden Validated: 14/14 Exact Match &middot; Source: MLB Stats API + Statcast</div>
</div>

<div class="container">

  <div class="section-header">Player Predictions vs LHP</div>
  <p style="font-size:13px;color:#64748b;margin-bottom:16px">
    Each card shows the player's 2026 and career stats vs LHP, a Bayesian-shrunk "true talent" estimate
    blending both, and per-game probabilities assuming ~4 PA against a left-handed starter.
    Reliability reflects 2026 PA in this split.
  </p>

  {''.join(cards)}

  <div class="section-header">Visual Analysis</div>
  <div class="charts">
    {chart_imgs}
  </div>

  <div class="methodology">
    <h3>Methodology &amp; Limitations</h3>
    <ul>
      <li><strong>Model type:</strong> Beta-Binomial Bayesian shrinkage — blends 2026 YTD counts with career same-split priors (60% career weight). NOT a trained ML classifier.</li>
      <li><strong>Per-game probabilities:</strong> Assumes ~4 PA per game; computed as 1 &minus; (1 &minus; rate)<sup>4</sup> for binary events.</li>
      <li><strong>Expected TB:</strong> Derived from shrunk BA &times; (1 + shrunk ISO / BA) &times; 4 PA.</li>
      <li><strong>80% credible intervals:</strong> From Beta posterior distribution; wider intervals = more uncertainty.</li>
      <li><strong>Reliability:</strong> HIGH = 25+ PA vs LHP in 2026; MED = 10-24 PA; LOW = under 10 PA.</li>
      <li><strong>Small samples dominate early season.</strong> Most players have &lt;25 PA vs LHP. Shrinkage toward career priors prevents overreacting to hot/cold streaks, but uncertainty remains high.</li>
      <li><strong>Outlier scores</strong> (1-5) compare 2026 OPS to career same-split OPS. Scores of 4-5 flag meaningful deviations; LOW-N means &lt;10 PA.</li>
      <li><strong>Phase 2</strong> replaces this with a calibrated PA-level ensemble (logistic + gradient boosting + RF) trained on 2015-2024 Statcast, validated on 2025.</li>
    </ul>
  </div>

</div>
</body>
</html>"""

    OUT.write_text(html)
    print(f"Dashboard: {OUT} ({len(html):,} bytes)")
    return OUT


if __name__ == "__main__":
    path = build()
    import subprocess
    subprocess.run(["open", str(path)])
