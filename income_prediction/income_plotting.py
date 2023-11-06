import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

from income_prediction.income import income_dataframe, income_dataframe_dropped, opt_random_forest_predictions, test_y

# Correlation Heatmap for all columns:
sns.heatmap(income_dataframe.corr(), annot=False, cmap="coolwarm")
plt.show()

# Correlation Heatmap for non dropped columns:
sns.heatmap(income_dataframe_dropped.corr(), annot=False, cmap="coolwarm")
plt.show()

# Plot the confusion matrix, for optimised random forest classifier:
true_income_values = test_y
predicted_income_values = opt_random_forest_predictions
confusion = confusion_matrix(test_y, predicted_income_values)
sns.heatmap(confusion, annot=True, fmt="d", cmap='Blues', xticklabels=['<=50K', '>50K'], yticklabels=['<=50K', '>50K'])
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.show()
