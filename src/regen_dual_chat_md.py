import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

def main():
    pa_path = ROOT / "data/raw/statcast_pa_level_league.parquet"
    df = pd.read_parquet(pa_path, columns=["batter", "pitcher", "events"])
    df["is_xbh"] = df["events"].isin({"double", "triple", "home_run"})
    n_xbh = (
        df.groupby(["batter", "pitcher"], as_index=False)["is_xbh"].sum().rename(columns={"is_xbh": "n_xbh"})
    )
    xbh_lookup = {(int(r.batter), int(r.pitcher)): int(r.n_xbh) for r in n_xbh.itertuples()}

    def norm_label(lab):
        if not lab:
            return "Low"
        s = str(lab).strip()
        return "Low" if s == "Very Low" else s

    def match_key(r):
        return (r["batter_name"], r["pitcher_name"], r.get("batter_team"), r.get("pitcher_team"))

    def vs_sp(r):
        bvp = r.get("bvp_career_vs_pitcher") or {}
        pa = int(bvp.get("bvp_pa") or 0)
        if pa <= 0:
            return "—"
        h = int(bvp.get("bvp_hits") or 0)
        hr = int(bvp.get("bvp_hr") or 0)
        bid, pid = r.get("batter_mlbam_id"), r.get("pitcher_mlbam_id")
        nx = xbh_lookup.get((int(bid), int(pid)))
        if nx is None:
            mbf = r.get("bvp_model_features") or {}
            nx = int(round(mbf.get("bvp_xbh_rate", 0.079) * mbf.get("bvp_pa_count", pa)))
        nx = max(nx, hr)
        if h:
            nx = min(nx, h)
        return f"{h}/{pa}, XB{nx}, HR{hr}"

    def row_lbl(r):
        return f"{r['batter_name']} ({r['batter_team']}) vs {r['pitcher_name']}"

    def pack(pr, er, which):
        if which == "hr":
            p3a, p3b = 100 * (1 - (1 - pr["p_hr"]) ** 3), 100 * (1 - (1 - er["p_hr"]) ** 3)
            ap_a, ap_b = pr["adj_p_hr"] * 100, er["adj_p_hr"] * 100
            ca, cb = pr["conf_hr"], er["conf_hr"]
            la, lb = norm_label(pr.get("conf_hr_label")), norm_label(er.get("conf_hr_label"))
        elif which == "hit":
            p3a, p3b = 100 * (1 - (1 - pr["p_hit"]) ** 3), 100 * (1 - (1 - er["p_hit"]) ** 3)
            ap_a, ap_b = pr["adj_p_hit"] * 100, er["adj_p_hit"] * 100
            ca, cb = pr["conf_hit"], er["conf_hit"]
            la, lb = norm_label(pr.get("conf_hit_label")), norm_label(er.get("conf_hit_label"))
        else:
            p3a, p3b = 100 * (1 - (1 - pr["p_xbh"]) ** 3), 100 * (1 - (1 - er["p_xbh"]) ** 3)
            ap_a, ap_b = pr["adj_p_xbh"] * 100, er["adj_p_xbh"] * 100
            ca, cb = pr["conf_xbh"], er["conf_xbh"]
            la, lb = norm_label(pr.get("conf_xbh_label")), norm_label(er.get("conf_xbh_label"))
        return ap_a, p3a, ca, la, ap_b, p3b, cb, lb

    P = json.loads((ROOT / "data/reports/todays_matchup_predictions.json").read_text())
    exp = {match_key(x): x for x in json.loads((ROOT / "data/reports/todays_matchup_predictions_exp.json").read_text())}
    by_key = {match_key(r): r for r in P}

    def emit_sec(num, name, rows, w):
        lines = [
            f"## Section {num}: {name}",
            "",
            "| # | Model 1 (prod) batter vs pitcher | Adj P | 3-PA | Conf | Label | vs SP | Model 2 (exp) batter vs pitcher | Adj P | 3-PA | Conf | Label | vs SP |",
            "|---:|---|---:|---:|---:|---|---|---|---:|---:|---:|---|",
        ]
        for i, pr in enumerate(rows, 1):
            er = exp[match_key(pr)]
            a = pack(pr, er, w)
            lines.append(
                f"| {i} | {row_lbl(pr)} | {a[0]:.1f}% | {a[1]:.1f}% | {a[2]:.3f} | {a[3]} | {vs_sp(pr)} | "
                f"{row_lbl(er)} | {a[4]:.1f}% | {a[5]:.1f}% | {a[6]:.3f} | {a[7]} | {vs_sp(er)} |"
            )
        return "\n".join(lines) + "\n\n"

    def emit_sec_compact(num, name, rows, w):
        """Narrow chat-friendly table: 5 columns (matchup once, prod & exp summaries)."""
        lines = [
            f"## Section {num}: {name}",
            "",
            "Each cell: **Adj P · 3-PA% · conf · label** (same batter–pitcher row for prod vs exp).",
            "",
            "| # | Matchup | Production | Experiment | vs SP |",
            "|---:|---|---|---|---|",
        ]
        for i, pr in enumerate(rows, 1):
            er = exp[match_key(pr)]
            a = pack(pr, er, w)
            prod_s = f"{a[0]:.1f}% · {a[1]:.1f}% · {a[2]:.2f} · {a[3]}"
            exp_s = f"{a[4]:.1f}% · {a[5]:.1f}% · {a[6]:.2f} · {a[7]}"
            lines.append(f"| {i} | {row_lbl(pr)} | {prod_s} | {exp_s} | {vs_sp(pr)} |")
        return "\n".join(lines) + "\n\n"

    S1 = sorted(P, key=lambda r: float(r.get("p_hr") or r.get("adj_p_hr") or 0), reverse=True)[:25]
    S2 = sorted(P, key=lambda r: float(r.get("p_hit") or r.get("adj_p_hit") or 0), reverse=True)[:25]
    S3 = sorted(P, key=lambda r: float(r.get("p_xbh") or r.get("adj_p_xbh") or 0), reverse=True)[:25]
    S4 = [r for r in P if (r.get("bvp_model_features") or {}).get("bvp_hr_count", 0) > 0]
    S4 = sorted(
        S4, key=lambda r: ((r.get("bvp_model_features") or {}).get("bvp_hr_count", 0), float(r.get("p_hr") or r.get("adj_p_hr") or 0)), reverse=True
    )[:25]
    S5 = [
        r
        for r in P
        if (r.get("bvp_model_features") or {}).get("bvp_pa_count", 0) >= 5
        and (r.get("bvp_career_vs_pitcher") or {}).get("bvp_ba", 0) >= 0.25
    ]
    S5 = sorted(
        S5, key=lambda r: ((r.get("bvp_career_vs_pitcher") or {}).get("bvp_hits", 0), float(r.get("p_hit") or r.get("adj_p_hit") or 0)), reverse=True
    )[:25]
    S6_ = []
    for r in P:
        bid, pid = r.get("batter_mlbam_id"), r.get("pitcher_mlbam_id")
        if bid is None or pid is None:
            continue
        if xbh_lookup.get((int(bid), int(pid)), 0) > 0:
            S6_.append(r)
    S6 = sorted(
        S6_,
        key=lambda r: (
            xbh_lookup.get((int(r["batter_mlbam_id"]), int(r["pitcher_mlbam_id"])), 0),
            float(r.get("p_xbh") or r.get("adj_p_xbh") or 0),
        ),
        reverse=True,
    )[:25]

    chunks = [
        emit_sec(1, "Top 25 — Home Runs (prod order; same matchup in both columns)", S1, "hr"),
        emit_sec(2, "Top 25 — Hits", S2, "hit"),
        emit_sec(3, "Top 25 — XBH", S3, "xbh"),
        emit_sec(4, "BvP-Favorable — Home Runs (BvP HR > 0)", S4, "hr"),
        emit_sec(5, "BvP-Favorable — Hits (PA ≥ 5, BvP BA ≥ .250)", S5, "hit"),
        emit_sec(6, "BvP-Favorable — XBH (career BvP XBH > 0 in Statcast)", S6, "xbh"),
    ]

    def dedicated(which, key, title):
        cfn = {"hr": "conf_hr", "hit": "conf_hit", "xbh": "conf_xbh"}[which]
        Pp = sorted(P, key=lambda r: r[key], reverse=True)[:25]
        Pe = sorted(P, key=lambda r: exp[match_key(r)][key], reverse=True)[:25]
        kp, ke = {match_key(x) for x in Pp}, {match_key(x) for x in Pe}
        ov = kp & ke
        gaps = [abs(by_key[x][key] - exp[x][key]) for x in ov]
        mean = 100 * sum(gaps) / len(gaps) if gaps else 0.0
        lines = [
            f"### Dedicated Top 25 — {title}",
            "",
            "| # | Model 1 (prod) batter | Conf | vs SP | Model 2 (exp) batter | Conf | vs SP |",
            "|---:|---|---:|---|---|---|---:|---|",
        ]
        for i in range(25):
            a, b = Pp[i], Pe[i]
            e_row = exp[match_key(b)]
            lines.append(
                f"| {i + 1} | {row_lbl(a)} | {a[cfn]:.3f} | {vs_sp(a)} | {row_lbl(b)} | {e_row[cfn]:.3f} | {vs_sp(e_row)} |"
            )
        lines.append("")
        lines.append(
            f"**Comparison:** Overlap: **{len(ov)}** | Prod-only: **{len(kp - ke)}** | Exp-only: **{len(ke - kp)}** | "
            f"Mean |Δ P| on overlap: **{mean:.2f}** pts"
        )
        lines.append("")
        return "\n".join(lines) + "\n"

    chunks.extend(
        [
            dedicated("hr", "p_hr", "Home Runs (rank by raw P(HR) within each model)"),
            dedicated("hit", "p_hit", "Hits (rank by raw P(Hit) within each model)"),
            dedicated("xbh", "p_xbh", "XBH (rank by raw P(XBH) within each model)"),
        ]
    )
    out = ROOT / "data/reports/dual_model_chat_tables_2026-04-22.md"
    out.write_text("".join(chunks))
    print(f"Wrote {out.relative_to(ROOT)}")

    compact_chunks = [
        "# Dual model — compact tables (slate 2026-04-22)\n\n",
        emit_sec_compact(1, "Top 25 — Home Runs (prod rank)", S1, "hr"),
        emit_sec_compact(2, "Top 25 — Hits (prod rank)", S2, "hit"),
        emit_sec_compact(3, "Top 25 — XBH (prod rank)", S3, "xbh"),
        emit_sec_compact(4, "BvP-favorable — HR (BvP HR > 0)", S4, "hr"),
        emit_sec_compact(5, "BvP-favorable — Hits (PA ≥ 5, BvP BA ≥ .250)", S5, "hit"),
        emit_sec_compact(6, "BvP-favorable — XBH (Statcast BvP XBH > 0)", S6, "xbh"),
    ]
    compact_out = ROOT / "data/reports/dual_model_chat_tables_compact_2026-04-22.md"
    compact_out.write_text("".join(compact_chunks))
    print(f"Wrote {compact_out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
