import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

from income_prediction.income import income_dataframe_dropped, opt_random_forest_predictions, test_y

# Correlation Heatmap for attributes with the strongest correlation with income:
normalised_text_columns = [col.lower().replace("-", " ").replace("_", " ") for col in income_dataframe_dropped.columns]
income_dataframe_dropped.columns = normalised_text_columns
sns.heatmap(income_dataframe_dropped.corr(), annot=False, cmap="coolwarm")
plt.title('Correlation Heatmap of Income Attributes')
plt.show()

# Plot the confusion matrix for the optimised random forest classifier:
true_income_values = test_y
predicted_income_values = opt_random_forest_predictions
matrix = confusion_matrix(true_income_values, predicted_income_values)
sns.heatmap(matrix, annot=True, fmt="d", cmap='Blues', xticklabels=['<=$50k', '>$50k'], yticklabels=['<=$50k', '>$50k'])
plt.title('Confusion Matrix for Optimised Random Forest Classifier')
plt.xlabel('Predicted Income')
plt.ylabel('True Income')
plt.show()
