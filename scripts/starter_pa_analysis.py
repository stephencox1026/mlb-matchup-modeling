import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

df = pd.read_parquet('/Users/stephencox/Desktop/Cursor Projects/LAD vs LHP Analysis/data/raw/statcast_bulk/statcast_2024.parquet')

# ── 1. Isolate plate appearances (rows where events is non-null = end of PA)
pa = df[df['events'].notna()].copy()

# ── 2. Identify starting pitchers for each game-side
#    The starter is the pitcher who appears in inning 1 for each half (Top/Bot).
#    For Top innings, the pitcher is the home team's pitcher (fielding team).
#    For Bot innings, the pitcher is the away team's pitcher.
#    We need to map: game_pk + inning_topbot -> starting pitcher

inning1 = df[df['inning'] == 1]
starters = (
    inning1.groupby(['game_pk', 'inning_topbot'])['pitcher']
    .first()
    .reset_index()
    .rename(columns={'pitcher': 'starting_pitcher'})
)

print(f"Total games identified: {starters['game_pk'].nunique()}")
print(f"Total starter-game-sides: {len(starters)}")
print()

# ── 3. Merge starter info onto PAs
#    When inning_topbot == 'Top', the batting team is the away team and the
#    pitcher they face is the home starter (identified from inning_topbot='Top' in inning 1).
#    When inning_topbot == 'Bot', the batting team is the home team, facing
#    the away starter (identified from inning_topbot='Bot' in inning 1).

pa = pa.merge(starters, on=['game_pk', 'inning_topbot'], how='left')

# Flag PAs against the starter
pa['vs_starter'] = pa['pitcher'] == pa['starting_pitcher']

print(f"Total PAs in 2024: {len(pa):,}")
print(f"PAs vs starters: {pa['vs_starter'].sum():,}")
print(f"PAs vs relievers: {(~pa['vs_starter']).sum():,}")
print(f"Pct vs starters: {pa['vs_starter'].mean()*100:.1f}%")
print()

# ── 4. Determine batting order position (1-9) for each batter in each game-side
#    Use at_bat_number ordering to find the first 9 unique batters.
pa_sorted = pa.sort_values(['game_pk', 'inning_topbot', 'at_bat_number'])

def assign_lineup_pos(group):
    seen = []
    batter_to_pos = {}
    for batter in group['batter']:
        if batter not in batter_to_pos:
            pos = len(seen) + 1
            if pos <= 9:
                batter_to_pos[batter] = pos
                seen.append(batter)
            else:
                batter_to_pos[batter] = np.nan
    group = group.copy()
    group['lineup_pos'] = group['batter'].map(batter_to_pos)
    return group

pa_sorted = pa_sorted.groupby(['game_pk', 'inning_topbot'], group_keys=False).apply(assign_lineup_pos)

# Filter to only PAs vs starters
vs_starter = pa_sorted[pa_sorted['vs_starter']].copy()

print("="*70)
print("PLATE APPEARANCES PER BATTER VS THE STARTING PITCHER (2024 MLB)")
print("="*70)

# ── 5. Count PAs per batter per game-side vs the starter
batter_pa_counts = (
    vs_starter.groupby(['game_pk', 'inning_topbot', 'batter', 'lineup_pos'])
    .size()
    .reset_index(name='pa_vs_starter')
)

# ── 5a. Overall distribution
print("\n── OVERALL DISTRIBUTION (all batters vs starter) ──")
print(f"  Mean PAs vs starter:   {batter_pa_counts['pa_vs_starter'].mean():.2f}")
print(f"  Median PAs vs starter: {batter_pa_counts['pa_vs_starter'].median():.1f}")
print(f"  Std dev:               {batter_pa_counts['pa_vs_starter'].std():.2f}")
print(f"  Min:                   {batter_pa_counts['pa_vs_starter'].min()}")
print(f"  Max:                   {batter_pa_counts['pa_vs_starter'].max()}")
print()

# Distribution counts
print("  Distribution of PA counts vs starter:")
pa_dist = batter_pa_counts['pa_vs_starter'].value_counts().sort_index()
total_batters = len(batter_pa_counts)
for count, freq in pa_dist.items():
    pct = freq / total_batters * 100
    bar = '█' * int(pct / 2)
    print(f"    {count} PAs: {freq:>6,} batters ({pct:>5.1f}%) {bar}")

# ── 5b. By lineup position
print("\n── MEAN PAs VS STARTER BY LINEUP POSITION ──")
lineup_stats = (
    batter_pa_counts[batter_pa_counts['lineup_pos'].between(1, 9)]
    .groupby('lineup_pos')['pa_vs_starter']
    .agg(['mean', 'median', 'std', 'count'])
    .round(2)
)
print(lineup_stats.to_string())

# ── 5c. PA distribution by lineup position
print("\n── PA DISTRIBUTION BY LINEUP POSITION (% of batters getting N PAs) ──")
lineup_pa = batter_pa_counts[batter_pa_counts['lineup_pos'].between(1, 9)].copy()
cross = pd.crosstab(
    lineup_pa['lineup_pos'],
    lineup_pa['pa_vs_starter'],
    normalize='index'
) * 100
print(cross.round(1).to_string())

# ── 6. Starter workload analysis
print("\n" + "="*70)
print("STARTING PITCHER WORKLOAD (2024 MLB)")
print("="*70)

# For each game-side, find the max inning the starter pitched in
starter_pitches = df.merge(starters, on=['game_pk', 'inning_topbot'], how='inner')
starter_pitches = starter_pitches[starter_pitches['pitcher'] == starter_pitches['starting_pitcher']]

# Max inning pitched by starter per game-side
starter_innings = (
    starter_pitches.groupby(['game_pk', 'inning_topbot'])['inning']
    .max()
    .reset_index(name='last_inning')
)

print(f"\n── INNINGS PITCHED BY STARTERS ──")
print(f"  Mean last inning:   {starter_innings['last_inning'].mean():.2f}")
print(f"  Median last inning: {starter_innings['last_inning'].median():.1f}")
print(f"  Std dev:            {starter_innings['last_inning'].std():.2f}")
print()
print("  Distribution of last inning pitched:")
inn_dist = starter_innings['last_inning'].value_counts().sort_index()
total_starts = len(starter_innings)
for inn, freq in inn_dist.items():
    pct = freq / total_starts * 100
    bar = '█' * int(pct / 2)
    print(f"    Inning {inn}: {freq:>5,} starts ({pct:>5.1f}%) {bar}")

# Total batters faced by starter per game
starter_batters_faced = (
    vs_starter.groupby(['game_pk', 'inning_topbot'])
    .size()
    .reset_index(name='batters_faced')
)

print(f"\n── TOTAL BATTERS FACED BY STARTER (per game) ──")
print(f"  Mean:   {starter_batters_faced['batters_faced'].mean():.1f}")
print(f"  Median: {starter_batters_faced['batters_faced'].median():.1f}")
print(f"  Std:    {starter_batters_faced['batters_faced'].std():.1f}")
print()
print("  Distribution:")
bf_dist = starter_batters_faced['batters_faced'].value_counts().sort_index()
for bf, freq in bf_dist.items():
    pct = freq / total_starts * 100
    if pct >= 1.0:
        bar = '█' * int(pct / 2)
        print(f"    {bf:>2} BF: {freq:>5,} starts ({pct:>5.1f}%) {bar}")

# ── 7. Times through the order analysis
print(f"\n── TIMES THROUGH THE ORDER (n_thruorder_pitcher) FOR STARTERS ──")
starter_pa_thru = vs_starter['n_thruorder_pitcher'].value_counts().sort_index()
total_starter_pa = len(vs_starter)
for tto, freq in starter_pa_thru.items():
    pct = freq / total_starter_pa * 100
    print(f"    Pass {tto}: {freq:>6,} PAs ({pct:>5.1f}%)")

# How many starters get through each pass
starter_max_tto = (
    vs_starter.groupby(['game_pk', 'inning_topbot'])['n_thruorder_pitcher']
    .max()
    .reset_index(name='max_pass')
)
print(f"\n  How many starters reach each pass through the order:")
tto_dist = starter_max_tto['max_pass'].value_counts().sort_index()
for tto, freq in tto_dist.items():
    pct = freq / len(starter_max_tto) * 100
    print(f"    Reached pass {tto}: {freq:>5,} starts ({pct:>5.1f}%)")

print("\n" + "="*70)
print("KEY TAKEAWAY SUMMARY")
print("="*70)
mean_pa = batter_pa_counts['pa_vs_starter'].mean()
mode_pa = batter_pa_counts['pa_vs_starter'].mode().iloc[0]
pct_2 = (batter_pa_counts['pa_vs_starter'] == 2).mean() * 100
pct_3 = (batter_pa_counts['pa_vs_starter'] == 3).mean() * 100
pct_4 = (batter_pa_counts['pa_vs_starter'] == 4).mean() * 100
pct_5 = (batter_pa_counts['pa_vs_starter'] == 5).mean() * 100
pct_2_or_3 = pct_2 + pct_3
mean_bf = starter_batters_faced['batters_faced'].mean()
mean_inn = starter_innings['last_inning'].mean()

print(f"""
  - Average batter gets {mean_pa:.2f} PAs vs the starting pitcher
  - Most common: {mode_pa} PAs (mode)
  - {pct_2:.1f}% get exactly 2 PAs, {pct_3:.1f}% get 3, {pct_4:.1f}% get 4, {pct_5:.1f}% get 5
  - {pct_2_or_3:.1f}% of batters get 2 or 3 PAs vs the starter
  - Average starter faces {mean_bf:.1f} batters and pitches into inning {mean_inn:.1f}
""")

# Lineup-specific summary
print("  By lineup position (mean PAs vs starter):")
for pos in range(1, 10):
    row = lineup_stats.loc[pos]
    print(f"    #{pos}: {row['mean']:.2f} PAs")
