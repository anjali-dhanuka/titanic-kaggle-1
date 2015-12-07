import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.cross_validation import KFold
import preprocess_data as preproc
import numpy as np

predictors = ["Pclass", "Age", "SibSp", "Parch", "Fare", "Embarked"]

lin_reg = LinearRegression()

train_data = preproc.preprocess_silent("train")
test_data = preproc.preprocess_silent("test")

kfold = KFold(train_data.shape[0], n_folds=3, random_state=1)


def train_linreg():
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


def test_linreg():
    predictions = lin_reg.predict(test_data[predictors])

    predictions_int = []
    for prediction in predictions:
        if prediction < 0.5:
            predictions_int.append(0)
        else:
            predictions_int.append(1)

    submission = pd.DataFrame({
        "PassengerId": test_data["PassengerId"],
        "Survived": predictions_int
    })

    return submission


if __name__ == '__main__':
    regr_accuracy = train_linreg()
    print "Accuracy of prediction: %s" % regr_accuracy
    kaggle_submission = test_linreg()
    kaggle_submission.to_csv('bin/kaggle.csv', index=False)