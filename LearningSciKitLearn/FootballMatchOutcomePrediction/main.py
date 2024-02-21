import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score

from LearningSciKitLearn.FootballMatchOutcomePrediction.util import mapping, rolling_averages, make_predictions

# Get data from local csv file:
matches_df = pd.read_csv("matches.csv", index_col=0)

# Missing data from relegated teams, and liverpool 2021:
matches_per_team, teams, seasons = 38, 20, 2
total_matches_irl = matches_per_team * teams * seasons  # 1520
total_matches_in_data = matches_df["team"].value_counts()  # 1389

# Data (inc. target) must be numeric, to be used in machine learning models:
del matches_df["comp"]
del matches_df["notes"]
matches_df["date"] = pd.to_datetime(matches_df["date"])
matches_df["venue_code"] = matches_df["venue"].astype("category").cat.codes
matches_df["opp_code"] = matches_df["opponent"].astype("category").cat.codes
matches_df["hour"] = matches_df["time"].str.replace(":.+", "", regex=True).astype("int")
matches_df["day_code"] = matches_df["date"].dt.dayofweek
matches_df["target"] = (matches_df["result"] == "W").astype("int")

# Initialise sklearn random forest model:
random_forest_model_v1 = RandomForestClassifier(n_estimators=50, min_samples_split=10, random_state=1)

# Testing data must come after training data in time:
train = matches_df[matches_df["date"] <= "2022-03-01"]
test = matches_df[matches_df["date"] > "2022-03-01"]

# Fit model with chosen numerical columns:
predictor_columns = ["venue_code", "opp_code", "hour", "day_code"]
random_forest_model_v1.fit(train[predictor_columns], train["target"])
predictions = random_forest_model_v1.predict(test[predictor_columns])

# Get model accuracy & precision:
random_forest_model_accuracy = accuracy_score(test["target"], predictions)
print(f"random_forest_model_v1 accuracy: {random_forest_model_accuracy * 100:.2f}%")
random_forest_model_precision = precision_score(test["target"], predictions)
print(f"random_forest_model_v1 precision: {random_forest_model_precision * 100:.2f}%")

# Check prediction vs truth:
preds_and_targets = pd.DataFrame(dict(actual=test["target"], prediction=predictions))
preds_vs_truth = pd.crosstab(index=preds_and_targets["actual"], columns=preds_and_targets["prediction"])

# Create (3 week) shooting data rolling averages for each team:
cols = ["gf", "ga", "sh", "sot", "dist", "fk", "pk", "pkatt"]
new_cols = [f"{c}_rolling_averages" for c in cols]
matches_rolling_df = matches_df.groupby("team").apply(lambda x: rolling_averages(x, cols, new_cols))
matches_rolling_df.index = range(matches_rolling_df.shape[0])

# Merge combined_df and matches_rolling_df, into df that has prediction, truth and match details:
combined, precision = make_predictions(random_forest_model_v1, matches_rolling_df, predictor_columns + new_cols)
merge_columns = ["date", "team", "opponent", "result"]
preds_and_details = combined.merge(matches_rolling_df[merge_columns], left_index=True, right_index=True)
preds_and_details["new_team"] = preds_and_details["team"].map(mapping)

# Each game appears twice, once for home team, once for away team, merge to get one row per game:
left_on, right_on = ["date", "new_team"], ["date", "opponent"]
final_df = preds_and_details.merge(preds_and_details, left_on=left_on, right_on=right_on)

# When home team predicted to win & away team predicted to lose:
home_w_away_l = final_df[(final_df["predicted_x"] == 1) & (final_df["predicted_y"] == 0)]["actual_x"].value_counts()
home_l_away_w = final_df[(final_df["predicted_x"] == 0) & (final_df["predicted_y"] == 1)]["actual_x"].value_counts()
home_w_away_l_accuracy = home_w_away_l[1] / (home_w_away_l[1] + home_w_away_l[0])
home_l_away_w_accuracy = home_l_away_w[0] / (home_l_away_w[0] + home_l_away_w[1])
print(f"home_win_away_loss accuracy: {home_w_away_l_accuracy * 100:.2f}%")
print(f"home_loss_away_win accuracy: {home_l_away_w_accuracy * 100:.2f}%")
