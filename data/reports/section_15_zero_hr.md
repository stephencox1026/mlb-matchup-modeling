## Section 15: No HR Model — P(0 HR) per game

**NB dispersion k = 12.65** (fit on 2022-2024 bulk; Poisson would systematically under-predict zero-HR rate by ~3 pp).
**League HR/PA baseline = 0.0303**.  **Mean PA/team/game = 38**.

Park PF: two-sided HR factor from `data/priors/park_hr_factors.json` (L2). BF̂ for each starter conditioned on Section 9 projected runs (heavier projection → shorter outing → more PAs to bullpen). **Bullpen rate**: per-team HR/PA blend from `data/priors/team_bullpen_hr.json` (Beta-Binomial EB shrunk; CLE 2.44% to LAA 3.09%). **Weather**: live Open-Meteo forecast at stadium for first-pitch hour; multiplier captures temp + wind-projected-onto-CF (capped ±20%); indoor games get mult 1.0.

| Rank | Matchup | Park | PF | Temp | Wind | Wx Mult | Pen A | Pen H | λ_total | E[HR] | **P(0 HR)** | Away SP (BF̂) | Home SP (BF̂) |
|---:|---|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---|---|
| 1 | MIL @ STL | STL | 0.92 | 48°F | 6 mph cross | 0.907 | 2.73% | 2.49% | 1.33 | 1.33 | **28.2%** | Brandon Sproat (24) | Andre Pallante (24) |
| 2 | CLE @ KC | KC | 0.91 | 52°F | 1 mph calm | 0.922 | 2.44% | 2.60% | 1.43 | 1.43 | **25.9%** | Gavin Williams (24) | Stephen Kolek (24) |
| 3 | SD @ SF | SF | 0.81 | 60°F | 12 mph out_to_cf | 1.069 | 2.52% | 2.46% | 1.44 | 1.44 | **25.7%** | Walker Buehler (24) | Logan Webb (24) |
| 4 | PIT @ AZ | AZ | 0.88 | — | — | 1.000 | 2.46% | 2.88% | 1.52 | 1.52 | **23.9%** | Bubba Chandler (24) | Eduardo Rodriguez (24) |
| 5 | BAL @ MIA | MIA | 0.96 | 83°F | indoor | 1.000 | 2.61% | 2.71% | 1.59 | 1.59 | **22.4%** | Chris Bassitt (24) | Sandy Alcantara (24) |
| 6 | CIN @ CHC | CHC | 0.99 | 55°F | 3 mph calm | 0.945 | 2.88% | 2.89% | 1.67 | 1.67 | **20.9%** | Andrew Abbott (24) | Jameson Taillon (24) |
| 7 | TOR @ TB | TB | 1.03 | 82°F | indoor | 1.000 | 3.02% | 2.95% | 1.83 | 1.83 | **18.1%** | Kevin Gausman (24) | Drew Rasmussen (24) |
| 8 | BOS @ DET | DET | 0.98 | 57°F | 11 mph out_to_cf | 1.039 | 2.65% | 2.61% | 1.83 | 1.83 | **18.1%** | Jovani Morán (24) | Framber Valdez (24) |
| 9 | NYM @ COL | COL | 1.17 | 33°F | 9 mph in_from_cf | 0.836 | 2.74% | 2.89% | 1.84 | 1.84 | **17.9%** | Freddy Peralta (24) | Michael Lorenzen (24) |
| 10 | ATL @ SEA | SEA | 0.95 | 64°F | 5 mph out_to_cf | 1.010 | 2.77% | 2.75% | 1.84 | 1.84 | **17.9%** | Bryce Elder (24) | George Kirby (24) |
| 11 | CWS @ LAA | LAA | 1.10 | 65°F | 10 mph out_to_cf | 1.075 | 2.75% | 3.09% | 2.01 | 2.01 | **15.5%** | Erick Fedde (24) | Sam Aldegheri (24) |
| 12 | LAD @ HOU | HOU | 1.02 | 77°F | indoor | 1.000 | 2.67% | 2.81% | 2.09 | 2.09 | **14.5%** | Shohei Ohtani (24) | Peter Lambert (24) |
| 13 | MIN @ WSH | WSH | 1.07 | 84°F | 16 mph out_to_cf | 1.182 | 2.67% | 3.02% | 2.20 | 2.20 | **13.2%** | Taj Bradley (24) | Cade Cavalli (24) |
| 14 | ATH @ PHI | PHI | 1.10 | 82°F | 18 mph out_to_cf | 1.173 | 2.61% | 2.71% | 2.35 | 2.35 | **11.5%** | Luis Severino (24) | Cristopher Sánchez (24) |
| 15 | TEX @ NYY | NYY | 1.08 | 75°F | 13 mph out_to_cf | 1.141 | 2.87% | 2.48% | 2.40 | 2.40 | **11.1%** | Jacob deGrom (24) | Elmer Rodríguez (24) |

_Sorted by highest P(0 HR). Higher = better target for an Under HR / no-HR-game prop. Wx Mult > 1 = HR-friendly conditions; < 1 = HR-suppressing._