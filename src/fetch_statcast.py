"""Fetch 2026 YTD Statcast batted-ball data for the 14 Dodgers hitters."""
import time
import pandas as pd
from pybaseball import statcast_batter
from config import SEASON, RAW_DIR, MASTER_DIR, REPORTS_DIR

IDS_CSV = RAW_DIR / "player_ids.csv"
START_DATE = "2026-03-20"
END_DATE = "2026-12-31"


def fetch_statcast_batters() -> pd.DataFrame:
    ids = pd.read_csv(IDS_CSV)
    frames = []
    for _, player in ids.iterrows():
        pid = int(player["mlbam_id"])
        name = player["name"]
        try:
            df = statcast_batter(START_DATE, END_DATE, pid)
            if df is not None and len(df) > 0:
                df["player_name_clean"] = name
                frames.append(df)
                print(f"  {name}: {len(df)} pitches")
            else:
                print(f"  {name}: no data")
        except Exception as e:
            print(f"  {name}: ERROR - {e}")
        time.sleep(1.0)

    if not frames:
        print("WARNING: No Statcast data retrieved")
        return pd.DataFrame()

    return pd.concat(frames, ignore_index=True)


def compute_contact_summary(sc: pd.DataFrame) -> pd.DataFrame:
    """Aggregate Statcast data into per-batter contact quality summary."""
    bbe = sc[sc["type"] == "X"].copy()
    if bbe.empty:
        return pd.DataFrame()

    summary = bbe.groupby("player_name_clean").agg(
        BBE=("launch_speed", "count"),
        avg_EV=("launch_speed", "mean"),
        max_EV=("launch_speed", "max"),
        avg_LA=("launch_angle", "mean"),
        barrel_pct=("launch_speed_angle", lambda x: (x == 6).mean()),
        hard_hit_pct=("launch_speed", lambda x: (x >= 95).mean()),
        avg_distance=("hit_distance_sc", "mean"),
    ).reset_index()

    summary = summary.rename(columns={"player_name_clean": "name"})
    for col in ["avg_EV", "max_EV", "avg_LA", "avg_distance"]:
        summary[col] = summary[col].round(1)
    for col in ["barrel_pct", "hard_hit_pct"]:
        summary[col] = (summary[col] * 100).round(1)

    return summary.sort_values("avg_EV", ascending=False).reset_index(drop=True)


if __name__ == "__main__":
    print("Fetching Statcast 2026 YTD for 14 Dodgers...")
    sc = fetch_statcast_batters()

    if sc.empty:
        print("No Statcast data — skipping")
        exit(0)

    sc.to_parquet(RAW_DIR / "statcast_2026_ytd.parquet", index=False)
    print(f"\nSaved {len(sc)} pitch rows to statcast_2026_ytd.parquet")

    summary = compute_contact_summary(sc)
    if not summary.empty:
        summary.to_csv(REPORTS_DIR / "statcast_contact_summary.csv", index=False)
        print(f"\nContact quality summary ({len(summary)} batters):")
        print(summary.to_string(index=False))
    else:
        print("No batted-ball events found")
