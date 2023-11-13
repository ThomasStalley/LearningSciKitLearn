from pprint import pprint

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV

# Load in the income dataset:
income_dataframe = pd.read_csv("income.csv")

# There is a numerical education column, so we drop the text category column:
income_dataframe.drop("education", inplace=True, axis=1)

# Drop final weight column:
income_dataframe.drop("fnlwgt", inplace=True, axis=1)

# Convert binary text columns to binary number columns:
income_dataframe["gender"] = income_dataframe["gender"].apply(lambda x: 1 if x == "Male" else 0)
income_dataframe["income"] = income_dataframe["income"].apply(lambda x: 1 if x == ">50K" else 0)

# one hot encode categorical columns, and remove original column:
categorical_columns = ["occupation", "workclass", "marital-status", "race", "native-country", "relationship"]
for column in categorical_columns:
    one_hot_encoded_df = pd.get_dummies(income_dataframe[column]).add_prefix(f"{column}_")
    income_dataframe.drop(column, axis=1, inplace=True)
    income_dataframe = pd.concat([income_dataframe, one_hot_encoded_df], axis=1)

# Remove bottom 80% of columns, ranked by correlation to income:
correlations = income_dataframe.corr()["income"].abs()
sorted_correlations = correlations.sort_values()
num_cols_to_drop = int(0.8 * len(income_dataframe.columns))
cols_to_drop = sorted_correlations.iloc[:num_cols_to_drop].index
income_dataframe_dropped = income_dataframe.drop(cols_to_drop, axis=1)

# Split data into training and testing data, and then further split into x and y data:
train_dataframe, test_dataframe = train_test_split(income_dataframe, test_size=0.2)
train_x, train_y = train_dataframe.drop("income", axis=1), train_dataframe["income"]
test_x, test_y = test_dataframe.drop("income", axis=1), test_dataframe["income"]

# Initialise random forest classifier:
random_forest = RandomForestClassifier()
random_forest.fit(train_x, train_y)
random_forest_score = random_forest.score(test_x, test_y)
print(f"random_forest score: {random_forest_score * 100:.2f}%")

# # Grid search to find optimised model:
# param_grid = {
#     "n_estimators": [50, 100, 250],
#     "max_depth": [5, 10, 30, None],
#     "min_samples_split": [2, 4],
#     "max_features": ['sqft', 'log2'],
# }
# grid_search = GridSearchCV(estimator=RandomForestClassifier(), param_grid=param_grid, verbose=10)
# grid_search.fit(train_x, train_y)
#
# # Optimised random forest classifier:
# opt_random_forest = grid_search.best_estimator_
opt_random_forest = RandomForestClassifier(
    max_depth=30,
    max_features='log2',
    min_samples_split=4,
    n_estimators=250,
)
opt_random_forest.fit(train_x, train_y)
opt_random_forest_predictions = opt_random_forest.predict(test_x)
optimised_random_forest_score = opt_random_forest.score(test_x, test_y)
print(f"optimised_random_forest score: {optimised_random_forest_score * 100:.2f}%")

# Investigate importance of each attribute, for initial and then optimised random forest classifiers:
importances = dict(zip(random_forest.feature_names_in_, random_forest.feature_importances_))
sorted_importances = {k: v for k, v in sorted(importances.items(), key=lambda x: x[1], reverse=True)}

opt_importances = dict(zip(opt_random_forest.feature_names_in_, opt_random_forest.feature_importances_))
sorted_optimised_importances = {k: v for k, v in sorted(opt_importances.items(), key=lambda x: x[1], reverse=True)}
