import matplotlib.pyplot as plt
import seaborn as sns

from IrisClassification.iris import iris_dataframe, predictions_dataframe, regularization_dataframe

# Histograms:
measurements = ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']
melted_dataframe = iris_dataframe.melt(value_vars=measurements, var_name='Measurement', value_name='Size')
histogram_grid = sns.FacetGrid(melted_dataframe, col="Measurement", col_wrap=2, sharex=False, sharey=False)
histogram_grid.map(sns.histplot, 'Size')
plt.tight_layout()
plt.show()

# Relplots:
fig, axs = plt.subplots(2, 2, figsize=(12, 12))
axs[0, 0].set_title('Sepal Length vs Species')
axs[0, 1].set_title('Sepal Width vs Species')
axs[1, 0].set_title('Petal Length vs Species')
axs[1, 1].set_title('Petal Width vs Species')
sns.scatterplot(x="sepal length (cm)", y="target_name", data=iris_dataframe, ax=axs[0, 0])
sns.scatterplot(x="sepal width (cm)", y="target_name", data=iris_dataframe, ax=axs[0, 1])
sns.scatterplot(x="petal length (cm)", y="target_name", data=iris_dataframe, ax=axs[1, 0])
sns.scatterplot(x="petal width (cm)", y="target_name", data=iris_dataframe, ax=axs[1, 1])
plt.tight_layout()
plt.show()

# Pairplots:
sns.pairplot(iris_dataframe, hue="target_name")
plt.tight_layout()
plt.show()

# Cross validation:
data = predictions_dataframe
_, axs = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))
style = predictions_dataframe["correct_prediction"].apply(lambda x: "X" if not x else "o")

axs[0].set_title("Species Prediction")
axs[0].legend(title='Predicted Label', loc='upper left', bbox_to_anchor=(1, 1))
sns.scatterplot(ax=axs[0], x="petal length (cm)", y="petal width (cm)", data=data, hue="prediction_label", style=style)

axs[1].set_title("True Species")
axs[1].legend(title='True Label', loc='upper left', bbox_to_anchor=(1, 1))
sns.scatterplot(ax=axs[1], x="petal length (cm)", y="petal width (cm)", data=data, hue="target_name")

plt.tight_layout()
plt.show()

# Regularisation parameter:
sns.scatterplot(x="c_values", y="regular_model_scores", data=regularization_dataframe)
plt.show()
