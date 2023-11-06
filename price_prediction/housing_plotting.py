import matplotlib.pyplot as plt
import seaborn as sns

from price_prediction.housing import housing_data, train_data, y_test_data, random_forest_model_predictions


# Histogram grid of training housing data:
train_data.columns = [col.replace("_", " ").lower() for col in train_data.columns]
train_data.hist(bins=20, figsize=(20, 15), color='skyblue', edgecolor='black', grid=False)
plt.suptitle('Histogram Grid of Training Housing Data Features'.title(), fontsize=16)
plt.xlabel('value')
plt.ylabel('frequency')
plt.show()

# Heatmap of training housing data, displaying correlation between attributes:
plt.figure(figsize=(12, 10))
sns.heatmap(data=train_data.corr(), annot=True, cmap="YlGnBu", fmt=".2f")
plt.title('Correlation Heatmap of Housing Attributes'.title(), fontsize=14)
plt.show()

# Scatterplot of geographical location, with hue visualising house value:
plt.figure(figsize=(14, 10))
sns.scatterplot(x="latitude", y="longitude", data=train_data.rename(columns=str.lower), hue="median house value",
                palette="coolwarm", alpha=0.6, edgecolor=None)
plt.title('Geographical Distribution of Median House Value'.title(), fontsize=14)
plt.xlabel('latitude')
plt.ylabel('longitude')
plt.legend(title='Median House Value ($)'.title(), title_fontsize='13', fontsize='11')
plt.show()

# Plotting true values vs. predicted values:
plt.figure(figsize=(10, 6))
plt.scatter(y_test_data, random_forest_model_predictions, alpha=0.3, edgecolor='k')
plt.plot([y_test_data.min(), y_test_data.max()], [y_test_data.min(), y_test_data.max()], '--r', linewidth=2)
plt.xlabel('true values ($)')
plt.ylabel('predictions ($)')
plt.title('Comparison of True Prices vs. Predicted Prices'.title(), fontsize=14)
plt.show()

# Scatter plot to show house value is capped at $500,000:
plt.figure(figsize=(10, 6))
high_value_houses = housing_data[housing_data['median_house_value'] > 400000]
high_value_houses['median_house_value'].plot(style='o', color='tomato')
plt.ylabel('median house value ($)')
plt.title('Distribution of High-Value Houses (Median Value > $400,000)'.title(), fontsize=14)
plt.xlabel('house index')
plt.show()
