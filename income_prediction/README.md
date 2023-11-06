### Personal Income Prediction Project:

Goal:

- Using machine learning to predict a person's income, based on various personal attributes.

Personal data cols:

- age
- workclass
- fnlwgt
- education
- educational-num
- marital-status
- occupation
- relationship
- race
- gender
- capital-gain
- capital-loss
- hours-per-week
- native-country
- income

Notes:

- One hot encoding is very useful for converting one text category column into n binary columns.
- We can use pandas corr() to investigate correlations between all columns/attributes. Or simply one chosen column vs
  the rest (allowing us to drop columns with low correlation to target attribute).

Model Accuracies:

- x.

Source:

- Course: [Income Prediction Machine Learning Project in Python](https://www.youtube.com/watch?v=dhoKFqhVJu0)

- Data: [Adult income dataset](https://www.kaggle.com/datasets/wenruliu/adult-income-dataset)