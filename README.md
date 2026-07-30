# FIFA World Cup 2026 — Player Performance Rating Prediction

Predicting a per-match player performance rating from in-match statistics, using an ensemble of tree-based models (Random Forest, XGBoost, LightGBM, CatBoost) benchmarked against a Ridge regression baseline.

## Overview

- **Target:** `player_rating` — a per-match player performance score
- **Data:** simulated FIFA World Cup 2026 player performance stats (match-level, per player)
- **Models:** Ridge (baseline), Random Forest, XGBoost, LightGBM, CatBoost
- **Best model:** LightGBM — Test R² = 0.958, MAE = 0.352, RMSE = 0.650

## Pipeline

1. Parse `match_date`, derive `day_of_tournament` as a numeric feature
2. Drop leakage and identifier columns (see [Leakage Investigation](#leakage-investigation) below)
3. Binary-encode `preferred_foot`, one-hot encode `position` / `tournament_stage` / `match_result`
4. Log-transform `market_value_eur` to handle skew
5. Train/test split (80/20) on the row index, done *before* any encoding that depends on the training data
6. Frequency-encode high-cardinality categoricals (`nationality`, `opponent_team`, `stadium`, `team`) — fit on the training split only, unseen test categories map to 0
7. Cross-validate all models (10-fold) on the training set
8. Evaluate all models on the held-out test set
9. Inspect feature importances and correlations as a leakage sanity check

## Leakage Investigation

The dataset ships with several columns that are either direct components of `player_rating` or aggregates that include the current match — these were dropped outright (see `cols_to_drop` in `model.py` for the full list and reasoning per group).

After dropping the obvious leakage, an early run still showed a small number of features dominating Random Forest's feature importance (~85% combined), which is the same red flag pattern as genuine target leakage. Those columns (`defensive_actions`, `recoveries`) were also dropped.

A second pass then showed a *new* top feature, `aerial_duels_won`, at 44% importance — raising the same concern again. Rather than keep removing features one at a time indefinitely, this was tested directly: the entire correlated defensive-action cluster (`aerial_duels_won`, `tackles`, `interceptions`, `clearances`, `aerial_duels_lost`) was dropped at once and the model re-evaluated.

| | R² |
|---|---|
| Full feature set | 0.958 |
| Without defensive cluster | 0.912 |

**Interpretation:** if these features were true leakage (i.e. effectively encoding the target), removing them would collapse R² toward 0. Instead it dropped moderately, from 0.958 to 0.912 — the signature of genuinely informative, mutually-correlated features (players active in one defensive category tend to be active in others), not a hidden formula. This matches domain expectations: a real match rating should reasonably correlate with a broad mix of defensive, attacking, and physical actions, which is exactly the smooth, gradually-declining correlation profile observed (top correlation 0.56, no single dominant feature) rather than the abrupt single-feature dominance seen in the original leakage case.

The features were kept in the final model.

## Baseline Comparison

A `RidgeCV` model (alpha selected via efficient leave-one-out CV) was added as a linear baseline:

| Model | CV R² | Test R² | Test MAE | Test RMSE |
|---|---|---|---|---|
| **LightGBM** | 0.9583 | **0.9577** | 0.352 | 0.650 |
| Random Forest | 0.9576 | 0.9560 | 0.359 | 0.663 |
| XGBoost | 0.9446 | 0.9452 | 0.419 | 0.740 |
| CatBoost | 0.9446 | 0.9443 | 0.426 | 0.746 |
| Ridge (baseline) | 0.6873 | 0.6897 | 1.423 | 1.761 |

The large gap between Ridge (~0.69) and the tree ensembles (~0.94–0.96) indicates the tree models are capturing genuine nonlinear structure and feature interactions — not simply overfitting to noise a linear model could also exploit.

Cross-validation and holdout test scores match closely across every model (within ~0.002 R²), confirming no leakage between CV folds or the train/test split.

## Results

**LightGBM** is the best-performing model on the held-out test set, narrowly ahead of Random Forest, with XGBoost and CatBoost trailing modestly behind.

## Possible Extensions

- Hyperparameter tuning for XGBoost/CatBoost (currently using fixed, untuned parameters)
- Replace frequency encoding with smoothed target encoding for high-cardinality categoricals (would require careful out-of-fold encoding to avoid reintroducing leakage)
- SHAP-based feature importance for a more rigorous alternative to Random Forest's built-in importances

## Files

- `pipeline.py` — full pipeline: data prep, leakage investigation, model training, evaluation
- `fifa_world_cup_2026_player_performance.csv` — input data, not included in repo. Download from [Kaggle: FIFA World Cup 2026 Player Performance Dataset](https://www.kaggle.com/datasets/rauffauzanrambe/fifa-world-cup-2026-player-performance-dataset) and place it in the project root before running `pipeline.py`
