"""Load batter lineup slot rollup (last 10 vs opposing starter hand) for inference."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from config import RAW_DIR


def load_lineup_slot_lookup(
    path: Path | None = None,
) -> dict[tuple[int, str], dict]:
    """
    Key: (batter_mlbam_id, opp_starter_throws) with throws 'L' or 'R'.
    Value: split label, n_games, median_slot, mode_slot, slots_json.
    """
    path = path or (RAW_DIR / "batter_lineup_spot_last10_vs_hand.parquet")
    if not path.is_file():
        return {}
    try:
        df = pd.read_parquet(path)
    except Exception:
        return {}
    if df.empty:
        return {}
    out: dict[tuple[int, str], dict] = {}
    for _, r in df.iterrows():
        oh = str(r["opp_hand"]).upper()[:1]
        if oh not in ("L", "R"):
            oh = "R"
        key = (int(r["batter_id"]), oh)
        out[key] = {
            "split": str(r.get("split") or ("vs_lhp" if oh == "L" else "vs_rhp")),
            "n_games": int(r["n_games"]),
            "median_slot": int(r["median_slot"]),
            "mode_slot": int(r["mode_slot"]),
            "slots_json": r.get("slots_json"),
        }
    return out
