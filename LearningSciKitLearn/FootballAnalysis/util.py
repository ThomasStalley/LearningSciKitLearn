import pandas as pd
from sklearn.metrics import precision_score


class MissingDict(dict):
    """Ensure club has same name for home games and away games."""
    __missing__ = lambda self, key: key


map_values = {
    "Brighton and Hove Albion": "Brighton",
    "Manchester United": "Manchester Utd",
    "Newcastle United": "Newcastle Utd",
    "Tottenham Hotspur": "Tottenham",
    "West Ham United": "West Ham",
    "Wolverhampton Wanderers": "Wolves"
}
mapping = MissingDict(**map_values)


def rolling_averages(group, columns, new_columns):
    """Compute rolling averages, for chosen columns, for 3 weeks - must use knowledge from past."""
    group = group.sort_values("date")
    rolling_stats = group[columns].rolling(3, closed='left').mean()
    group[new_columns] = rolling_stats
    group = group.dropna(subset=new_columns)
    return group


def make_predictions(model, data, predictors):
    """Train and test a mode, return prediction_vs_truth dataframe, and precision score."""
    train = data[data["date"] < '2022-01-01']
    test = data[data["date"] > '2022-01-01']
    model.fit(train[predictors], train["target"])
    preds = model.predict(test[predictors])
    combined = pd.DataFrame(dict(actual=test["target"], predicted=preds), index=test.index)
    precision = precision_score(test["target"], preds)
    return combined, precision
