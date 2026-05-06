# LAD @ COL — model top 10 (both teams) + Coors park-table adjustments

**Source:** `data/reports/todays_matchup_predictions.json`.

## Park-table methodology

**Stadium table (Coors, LAD@COL):** HR +11%, 2B/3B +33%, singles +11%.
Decompose `P(hit)` into singles `P(hit)−P(XBH)`, non-HR XBH `P(XBH)−P(HR)`, and HR `P(HR)`; apply row-specific K; cap and re-sum.

**Graphic key + slate note:** upper-50s (cold band vs carry), **red H** (dry → worse carry), mixed wind, Ballpark Pal ‘pitcher-friendly evening’.
**CARRY_DAMP** = 0.8852 multiplies the increment `(K−1)` for each stadium factor → effective **M_HR=1.0974**, **M_2B3B=1.2921**, **M_1B=1.0974**.

`Adj P` after park table = model `Adj P` × (park row raw / model raw) so confidence tier is unchanged.

## LAD batters

### Home run — rank by model Adj P(HR)

| # | Batter | vs SP | Raw P | Adj P (model) | P (park) | Adj P (park) | Conf | Grade |
|---:|---|---:|---:|---:|---:|---:|---:|---|

### Home run — rank by park-table Adj P(HR)

| # | Batter | vs SP | Raw P | Adj P (model) | P (park) | Adj P (park) | Conf | Grade |
|---:|---|---:|---:|---:|---:|---:|---:|---|

### Hit — rank by model Adj P(Hit)

| # | Batter | vs SP | Raw P | Adj P (model) | P (park) | Adj P (park) | Conf | Grade |
|---:|---|---:|---:|---:|---:|---:|---:|---|

### Hit — rank by park-table Adj P(Hit)

| # | Batter | vs SP | Raw P | Adj P (model) | P (park) | Adj P (park) | Conf | Grade |
|---:|---|---:|---:|---:|---:|---:|---:|---|

### XBH — rank by model Adj P(XBH)

| # | Batter | vs SP | Raw P | Adj P (model) | P (park) | Adj P (park) | Conf | Grade |
|---:|---|---:|---:|---:|---:|---:|---:|---|

### XBH — rank by park-table Adj P(XBH)

| # | Batter | vs SP | Raw P | Adj P (model) | P (park) | Adj P (park) | Conf | Grade |
|---:|---|---:|---:|---:|---:|---:|---:|---|

## COL batters

### Home run — rank by model Adj P(HR)

| # | Batter | vs SP | Raw P | Adj P (model) | P (park) | Adj P (park) | Conf | Grade |
|---:|---|---:|---:|---:|---:|---:|---:|---|

### Home run — rank by park-table Adj P(HR)

| # | Batter | vs SP | Raw P | Adj P (model) | P (park) | Adj P (park) | Conf | Grade |
|---:|---|---:|---:|---:|---:|---:|---:|---|

### Hit — rank by model Adj P(Hit)

| # | Batter | vs SP | Raw P | Adj P (model) | P (park) | Adj P (park) | Conf | Grade |
|---:|---|---:|---:|---:|---:|---:|---:|---|

### Hit — rank by park-table Adj P(Hit)

| # | Batter | vs SP | Raw P | Adj P (model) | P (park) | Adj P (park) | Conf | Grade |
|---:|---|---:|---:|---:|---:|---:|---:|---|

### XBH — rank by model Adj P(XBH)

| # | Batter | vs SP | Raw P | Adj P (model) | P (park) | Adj P (park) | Conf | Grade |
|---:|---|---:|---:|---:|---:|---:|---:|---|

### XBH — rank by park-table Adj P(XBH)

| # | Batter | vs SP | Raw P | Adj P (model) | P (park) | Adj P (park) | Conf | Grade |
|---:|---|---:|---:|---:|---:|---:|---:|---|
