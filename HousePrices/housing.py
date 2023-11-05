import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Get our housing data as pandas dataframe:
housing_data = pd.read_csv("housing.csv")
housing_data.dropna(inplace=True)

# Split housing data into training and testing data:
x_housing_data = housing_data.drop(["median_house_value"], axis=1)
y_housing_data = housing_data["median_house_value"]
x_train, x_test, y_train, y_test = train_test_split(x_housing_data, y_housing_data, test_size=0.2)
train_data = x_train.join(y_train)
test_data = x_test.join(y_test)

# Apply log transformation to skewed data columns:
train_data["total_rooms"] = np.log(train_data["total_rooms"] + 1)
train_data["total_bedrooms"] = np.log(train_data["total_bedrooms"] + 1)
train_data["population"] = np.log(train_data["population"] + 1)
train_data["households"] = np.log(train_data["households"] + 1)

test_data["total_rooms"] = np.log(test_data["total_rooms"] + 1)
test_data["total_bedrooms"] = np.log(test_data["total_bedrooms"] + 1)
test_data["population"] = np.log(test_data["population"] + 1)
test_data["households"] = np.log(test_data["households"] + 1)

# Use one hot encoding to transform our ocean_proximity text col into five boolean cols:
one_hot_encoded_ocean_dataframe = pd.get_dummies(train_data["ocean_proximity"])
train_data.drop("ocean_proximity", axis=1, inplace=True)
train_data = train_data.join(one_hot_encoded_ocean_dataframe)

one_hot_encoded_ocean_test_dataframe = pd.get_dummies(test_data["ocean_proximity"])
test_data.drop("ocean_proximity", axis=1, inplace=True)
test_data = test_data.join(one_hot_encoded_ocean_test_dataframe)

# Feature engineering: Add two new columns that hold potentially important data:
train_data["bedroom_ratio"] = train_data["total_bedrooms"] / train_data["total_rooms"]
train_data["household_rooms"] = train_data["total_rooms"] / train_data["households"]

test_data["bedroom_ratio"] = test_data["total_bedrooms"] / test_data["total_rooms"]
test_data["household_rooms"] = test_data["total_rooms"] / test_data["households"]

# Final data to train, then test our model:
x_train_data = train_data.drop(["median_house_value"], axis=1)
y_train_data = train_data["median_house_value"]

x_test_data = test_data.drop(["median_house_value"], axis=1)
y_test_data = test_data["median_house_value"]

# Scale x training and testing data:
scaler = StandardScaler()
scaled_x_train_data = scaler.fit_transform(x_train_data)
scaled_x_test_data = scaler.transform(x_test_data)

# Linear Regression model:
linear_regression_model = LinearRegression()
linear_regression_model.fit(x_train_data, y_train_data)
linear_regression_model_score = linear_regression_model.score(x_test_data, y_test_data)
print(f"linear_regression_model score: {100 * linear_regression_model_score:.2f}%")

scaled_linear_regression_model = LinearRegression()
scaled_linear_regression_model.fit(scaled_x_train_data, y_train_data)
scaled_linear_regression_model_score = scaled_linear_regression_model.score(scaled_x_test_data, y_test_data)
print(f"scaled_linear_regression_model score: {100 * scaled_linear_regression_model_score:.2f}%")

# Random Forest Regression model:
random_forest_model = RandomForestRegressor()
random_forest_model.fit(x_train_data, y_train_data)
random_forest_model_predictions = random_forest_model.predict(x_test_data)
random_forest_model_score = random_forest_model.score(x_test_data, y_test_data)
print(f"random_forest_model score: {100 * random_forest_model_score:.2f}%")

scaled_random_forest_model = RandomForestRegressor()
scaled_random_forest_model.fit(scaled_x_train_data, y_train_data)
scaled_random_forest_model_score = scaled_random_forest_model.score(scaled_x_test_data, y_test_data)
print(f"scaled_random_forest_model score: {100 * scaled_random_forest_model_score:.2f}%")

# Random Forest Regression model (median_house_value capped at 500000 in raw data):
clean_train_data = train_data[train_data['median_house_value'] < 5000000]
clean_x_train_data = clean_train_data.drop(["median_house_value"], axis=1)
clean_y_train_data = clean_train_data["median_house_value"]

clean_test_data = test_data[test_data['median_house_value'] < 5000000]
clean_x_test_data = clean_test_data.drop(["median_house_value"], axis=1)
clean_y_test_data = clean_test_data["median_house_value"]

clean_random_forest_model = RandomForestRegressor()
clean_random_forest_model.fit(clean_x_train_data, clean_y_train_data)
clean_random_forest_model_score = clean_random_forest_model.score(clean_x_test_data, clean_y_test_data)
print(f"clean_random_forest_model score: {100 * clean_random_forest_model_score:.2f}%")
