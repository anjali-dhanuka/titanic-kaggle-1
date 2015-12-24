import preprocess_data as preproc
import random_forests as randfor
import pandas as pd

predictors = ["Pclass", "Sex", "Fare", "Title"]


def generate_submission(test_data):
    predictions = randfor.algo.predict(test_data[predictors])

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
    titanic_train = preproc.preprocess_silent("train")
    titanic_train = randfor.add_feat_title(titanic_train)

    randfor.algo.fit(titanic_train[predictors], titanic_train["Survived"])

    titanic_test = preproc.preprocess_silent("test")
    titanic_test = randfor.add_feat_title(titanic_test)

    kaggle_submission = generate_submission(titanic_test)
    kaggle_submission.to_csv("bin/kaggle.csv", index=False)
