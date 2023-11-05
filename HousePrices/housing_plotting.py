import matplotlib.pyplot as plt
import seaborn as sns

from HousePrices.housing import housing_data, train_data, y_test_data, random_forest_model_predictions

# Histogram grid of training housing data:
train_data.hist()
plt.show()

# Heatmap of training housing data, displaying correlation between attributes:
sns.heatmap(data=train_data.corr(), annot=True, cmap="YlGnBu")
plt.show()

# Scatterplot of geographical location, with hue visualising house value:
sns.scatterplot(x="latitude", y="longitude", data=train_data, hue="median_house_value", palette="coolwarm")
plt.show()

# Plotting true values vs. predicted values:
plt.figure(figsize=(10, 6))
plt.scatter(y_test_data, random_forest_model_predictions, alpha=0.3)
plt.plot([y_test_data.min(), y_test_data.max()], [y_test_data.min(), y_test_data.max()], '--r', linewidth=2)
plt.xlabel('True Values')
plt.ylabel('Predictions')
plt.title('True Price vs. Predicted Price')
plt.show()

# Scatter plot to show house value is capped at $500,000:
high_value_houses = housing_data[housing_data['median_house_value'] > 400000]
high_value_houses['median_house_value'].plot(style='o')
plt.ylabel('Median House Value')
plt.title('Median House Value (> $400,000) vs. Index')
plt.show()
