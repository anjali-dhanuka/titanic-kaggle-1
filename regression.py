import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.cross_validation import KFold
import preprocess_data as preproc
import numpy as np

predictors = ["Pclass", "Age", "SibSp", "Parch", "Fare", "Embarked"]

lin_reg = LinearRegression()
log_reg = LogisticRegression(random_state=1)

train_data = preproc.preprocess_silent("train")
test_data = preproc.preprocess_silent("test")

kfold = KFold(train_data.shape[0], n_folds=3, random_state=1)


def fit_and_predict_linreg():
    predictions = []
    for train, test in kfold:
        train_predictors = train_data[predictors].iloc[train, :]
        train_target = train_data["Survived"].iloc[train]

        lin_reg.fit(train_predictors, train_target)
        test_predictions = lin_reg.predict(train_data[predictors].iloc[test, :])

        predictions.append(test_predictions)

        predictions = np.concatenate(predictions)
        predictions[predictions > .5] = 1
        predictions[predictions <=.5] = 0

        accuracy = sum(predictions[predictions == train_data["Survived"]]) / len(predictions)
    return accuracy


def fit_and_predict_logreg():
    log_reg.fit(train_data[predictors], train_data["Survived"])

    predictions = log_reg.predict(test_data[predictors])

    submission = pd.DataFrame({
        "PassengerId": test_data["PassengerId"],
        "Survived": predictions
    })

    return submission