
## Model and Results

**Dataset:** TMDB 5000 Movies (budget, runtime, release date, genres, language, production companies).

**Target:** worldwide box office revenue (USD).

**Features (all known before release):** budget, runtime, release month, genre, original language, production company.

**Model:** XGBoost regressor trained on a log-transformed target, wrapped with its preprocessing so the API returns dollars directly.

**Results (20% held-out test set):**

| Metric | Value |
|--------|-------|
| R2 | 0.41 |
| MAE | about $79M |
| Baseline MAE (always predict the average) | about $118M |

The model's error is roughly a third lower than the baseline.

**Limitations:**
- No cast, director, marketing spend or franchise data, so it cannot capture everything that drives revenue.
- Revenue is heavy-tailed, so the model tends to underpredict mega-hits.
- Only the 40 most common production companies are used; all others are grouped as "Other".

**Dataset history:** this project originally used a synthetic box office dataset where the model performed no better than predicting the average (R2 below 0). After diagnosing this, I switched to the real TMDB data, which gave a meaningful improvement over the baseline.
