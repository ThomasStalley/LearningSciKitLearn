import matplotlib.pyplot as plt
import seaborn as sns

from LearningSciKitLearn.SpeciesPrediction.iris import iris_dataframe, predictions_dataframe, regularization_dataframe


def format_column_name_lower(name):
    # This will format the column names to lowercase except for the first character
    return name.replace('_', ' ').replace('-', ' ').title().lower()


# Update dataframe columns to lowercase
iris_dataframe.columns = [format_column_name_lower(col) for col in iris_dataframe.columns]
predictions_dataframe.columns = [format_column_name_lower(col) for col in predictions_dataframe.columns]
regularization_dataframe.columns = [format_column_name_lower(col) for col in regularization_dataframe.columns]

# Histograms:
measurements = ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']
melted_dataframe = iris_dataframe.melt(value_vars=measurements, var_name='measurement', value_name='size')
histogram_grid = sns.FacetGrid(melted_dataframe, col="measurement", col_wrap=2, sharex=False, sharey=False)
histogram_grid.map(sns.histplot, 'size')
plt.tight_layout()
plt.show()

# Relplots:
fig, axs = plt.subplots(2, 2, figsize=(14, 10))
measurements = ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']
for ax, feature in zip(axs.flat, measurements):
    sns.scatterplot(x=feature, y="target name", data=iris_dataframe, ax=ax)
    ax.set_xlabel(feature.lower())
    ax.set_ylabel("target name")
    ax.tick_params(axis='both', which='major', labelsize=10)
plt.tight_layout()
plt.show()

# Pairplots:
sns.pairplot(iris_dataframe, hue="target name")
plt.tight_layout()
plt.show()

# Cross validation:
plt.rcParams["font.serif"] = ["Times New Roman"]
plt.rcParams["font.family"] = "serif"

_, axs = plt.subplots(nrows=1, ncols=2, figsize=(12, 6))
style = predictions_dataframe["correct prediction"].apply(lambda x: "X" if not x else "o")

sns.scatterplot(ax=axs[0], x="petal length (cm)", y="petal width (cm)", data=predictions_dataframe, hue="prediction label", style=style)
axs[0].set_title("Species Prediction", fontweight='bold', fontstyle='italic', fontsize=12)
axs[0].legend(title='Predicted Label', loc='upper left', bbox_to_anchor=(1, 1))

sns.scatterplot(ax=axs[1], x="petal length (cm)", y="petal width (cm)", data=predictions_dataframe, hue="target name")
axs[1].set_title("True Species", fontweight='bold', fontstyle='italic', fontsize=12)
axs[1].legend(title='True Label', loc='upper left', bbox_to_anchor=(1, 1))

plt.tight_layout()
plt.show()

# Regularisation parameter:
sns.scatterplot(x="c values", y="regular model scores", data=regularization_dataframe)
plt.title('Regularization Parameter vs Model Score')
plt.xlabel("c values")
plt.ylabel("regular model scores")
plt.tight_layout()
plt.show()
