import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict

# Create Iris dataframe:
species_mapping = {0: "setosa", 1: "versicolor", 2: "virginica"}
iris_dataset = datasets.load_iris()
iris_dataframe = pd.DataFrame(data=iris_dataset["data"], columns=iris_dataset["feature_names"])
iris_dataframe["target"] = iris_dataset["target"]
iris_dataframe['target_name'] = iris_dataframe['target'].map(species_mapping)

# Split data into training data and test data, then split test data into x and y:
iris_train_dataframe, iris_test_dataframe = train_test_split(iris_dataframe, test_size=0.25)
x_train = iris_train_dataframe.drop(columns=["target", "target_name"]).values
y_train = iris_train_dataframe["target"].values

# Basic single feature (petal length) model:
single_feature_model = np.vectorize(lambda petal_length: 0 if petal_length < 2.5 else 1 if petal_length < 4.8 else 2)
single_feature_predictions = np.array([single_feature_model(val) for val in x_train[:, 2]])
single_feature_prediction_accuracy = np.mean(single_feature_predictions == y_train)
single_feature_prediction_score = f"single_feature_model accuracy: {single_feature_prediction_accuracy * 100:.2f}%"
print(single_feature_prediction_score)

# Logistic regression model:
logistic_regression_model = LogisticRegression(max_iter=200)
xt, xv, yt, yv = train_test_split(x_train, y_train, test_size=0.25)
logistic_regression_model.fit(xt, yt)
y_predictions = logistic_regression_model.predict(xv)
logistic_regression_model_accuracy = np.mean(y_predictions == yv)
logistic_regression_model_score = f"logistic_regression_model accuracy: {logistic_regression_model_accuracy * 100:.2f}%"
print(logistic_regression_model_score)

# Use cross validation to evaluate our logistic regression model:
model_to_validate = LogisticRegression(max_iter=200)
cross_validation_scores = cross_val_score(model_to_validate, x_train, y_train, cv=5, scoring="accuracy")
cross_validation_accuracy = np.mean(cross_validation_scores)
cross_validation_score = f"logistic_regression_model cross validation score: {cross_validation_accuracy * 100:.2f}%"
print(cross_validation_score)

# Use cross validation to evaluate a random forest model:
random_forest_model = RandomForestClassifier()
random_forest_accuracies = cross_val_score(random_forest_model, x_train, y_train, cv=5, scoring="accuracy")
random_forest_accuracy = np.mean(random_forest_accuracies)
random_forest_score = f"random_forest_model cross validation score: {random_forest_accuracy * 100:.2f}%"
print(random_forest_score)

# Using final model on testing data:
finaL_logistic_regression_model = LogisticRegression(max_iter=200, C=5)
x_test = iris_test_dataframe.drop(columns=["target", "target_name"]).values
y_test = iris_test_dataframe["target"].values
finaL_logistic_regression_model.fit(x_train, y_train)
y_final_predictions = finaL_logistic_regression_model.predict(x_test)
final_model_accuracy = np.mean(y_final_predictions == y_test)
final_model_score = f"final_logistic_regression_model accuracy: {final_model_accuracy * 100:.2f}%\n"
print(final_model_score)

predictions_dataframe = iris_test_dataframe.copy()
predictions_dataframe["correct_prediction"] = (y_final_predictions == y_test)
predictions_dataframe["prediction"] = y_final_predictions
predictions_dataframe["prediction_label"] = predictions_dataframe["prediction"].map(species_mapping)

# Investigating regularization in our logistic regression model:
regular_model_scores = []
c_values = np.arange(0.02, 5.02, 0.02)
for c in c_values:
    regular_model = LogisticRegression(max_iter=300, C=c)
    regular_cross_validation_scores = cross_val_score(regular_model, x_train, y_train, cv=5, scoring="accuracy")
    regular_cross_validation_score = np.mean(regular_cross_validation_scores)
    regular_model_scores.append(regular_cross_validation_score)
regularization_dataframe = pd.DataFrame({"c_values": c_values, "regular_model_scores": regular_model_scores})
