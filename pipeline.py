import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from catboost import CatBoostRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, root_mean_squared_error


# ── Config ────────────────────────────────────────────────

config = {
    'random_state': 42,
    'test_size':    0.2,
    'n_splits':     10
}

# ── Base model param ─────────────────────────────────────

ridge_params = {'alphas': [0.1, 1.0, 10.0]}

# ── Manual/default model hyperparameters (untuned) ────────
rf_params = {
    'n_estimators': 300,
    'random_state': 42,
    'n_jobs':       -1,
}

xgb_params = {
    'n_estimators':  500,
    'learning_rate': 0.1,
    'max_depth':     6,
    'random_state':  42,
    'n_jobs':        -1,
    'objective':     'reg:squarederror',
}

lgbm_params = {
    'n_estimators':  500,
    'learning_rate': 0.05,
    'num_leaves':    31,
    'random_state':  42,
    'n_jobs':        -1,
    'verbose':       -1,
    'objective':     'regression',
}

cat_params = {
    'iterations':    500,
    'learning_rate': 0.1,
    'depth':         6,
    'random_state':  42,
    'verbose':       False,
    'loss_function': 'RMSE',
    'allow_writing_files': False,
}


df = pd.read_csv("fifa_world_cup_2026_player_performance.csv")

df["match_date"] = pd.to_datetime(df["match_date"])
df['day_of_tournament'] = (df['match_date'] - df['match_date'].min()).dt.days
df = df.drop(columns=["match_date"])

cols_to_drop = [

    # Leakage - match/tournament outcomes not known until after the match
    "goals_team",
    "goals_opponent",

    # Leakage - components of player_rating
    "performance_score",
    "offensive_contribution",
    "defensive_contribution",
    "possession_impact",
    "pressure_resistance",
    "creativity_score",
    "consistency_score",
    "clutch_performance_score",

    # Leakage - tournament aggregates (include current match)
    "total_goals_tournament",
    "total_assists_tournament",
    "total_minutes_tournament",
    "player_of_match_awards",
    "tournament_rating",

    # Suspected leakage - physical/fitness cluster, dominant correlation w/ target
    "top_speed_kmh",
    "distance_covered_km",
    "minutes_played",
    "sprint_distance_km",
    "accelerations",
    "decelerations",

    # Suspected leakage - dominates feature importance (~85% combined),
    # likely part of the target's construction formula
    "defensive_actions",
    "recoveries",

    # Identifiers - no predictive signal
    "player_id",
    "match_id",
    "player_name",

    # Redundant - covered by pass_accuracy
    "successful_passes",
    "total_passes",

    # Redundant - covered by save_percentage
    "saves",
    "goals_conceded",

    # Duplicate of stadium
    "city",

    # High cardinality, low relevance to national-team performance
    "club_name",

    # Raw string - replaced by day_of_tournament
    "jersey_number",

]

df = df.drop(columns=cols_to_drop)

df['preferred_foot'] = (df['preferred_foot'] == 'Right').astype(int)

df = pd.get_dummies(df, columns=['position', 'tournament_stage', 'match_result'], drop_first=True)

df["market_value_log"] = np.log1p(df["market_value_eur"])
df = df.drop(columns=["market_value_eur"])

# index-based split, done before encoding so no test info leaks in
train_idx, test_idx = train_test_split(
    df.index, random_state=config["random_state"], test_size=config["test_size"]
)

#* frequency encoding (fit on train only, applied to both splits)
for col in ["nationality", "opponent_team", "stadium", "team"]:
    freq = df.loc[train_idx, col].value_counts()
    df[col + '_freq'] = df[col].map(freq)
    # unseen categories in test won't be in `freq` -> map() gives NaN, fill with 0
    df[col + '_freq'] = df[col + '_freq'].fillna(0)
    df = df.drop(columns=[col])

#* defining X and y
X = df.drop(columns=["player_rating"])
y = df["player_rating"]

X_train, X_test = X.loc[train_idx], X.loc[test_idx]
y_train, y_test = y.loc[train_idx], y.loc[test_idx]


models = {
    "ridge": RidgeCV(**ridge_params),
    "rf": RandomForestRegressor(**rf_params),
    "catboost": CatBoostRegressor(**cat_params),
    "xgb": XGBRegressor(**xgb_params),
    "lgbm": LGBMRegressor(**lgbm_params)
}

# ── Cross-validated performance on the training set ───────
kfold = KFold(n_splits=config["n_splits"], random_state=config['random_state'], shuffle=True)

for name, model in models.items():
    scores = cross_val_score(model, X_train, y_train, cv=kfold, scoring="r2")
    print(f"{name}: R2 = {scores.mean():.4f} (+/- {scores.std():.4f})")

# ── Final holdout evaluation ───────────────────────────────
# Refit each model on the full training set and score it against
# X_test/y_test, which no model or CV fold has seen before.
# This confirms the CV scores above generalize and aren't the
# result of leakage between folds (e.g. via the frequency encoding).
test_results = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    test_results[name] = {
        "r2":   r2_score(y_test, y_pred),
        "mae":  mean_absolute_error(y_test, y_pred),
        "rmse": root_mean_squared_error(y_test, y_pred),
        "mse":  mean_squared_error(y_test, y_pred)
    }

# Collect into a DataFrame, best model (highest R2) on top
test_df = pd.DataFrame(test_results).T.sort_values("r2", ascending=False)

print("\n" + "=" * 50)
print("TEST RESULTS")
print("=" * 50)
print(test_df)

# ── Correlation check ──────────────────────────────────────
# Sanity check against the feature importances above: a smooth,
# spread-out correlation table (no single dominant feature) is
# consistent with a genuine composite rating rather than leakage.
print(df.corr(numeric_only=True)['player_rating'].sort_values(ascending=False).head(20))

# ── Feature importance check (quick leakage sanity check) ─

importances = pd.Series(models["rf"].feature_importances_, index=X_train.columns)
print(importances.sort_values(ascending=False).head(15))