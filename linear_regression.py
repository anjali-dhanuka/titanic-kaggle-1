from sklearn.linear_model import LinearRegression
from sklearn.cross_validation import KFold
import preprocess_data as preproc

predictors = ["Pclass", "Age", "SibSp", "Parch", "Fare", "Embarked"]

lin_reg = LinearRegression()
train_data = preproc.preprocess_silent()

kfold = KFold(train_data.shape[0], n_folds=3, random_state=1)


def hypothesis_fitting():
    predictions = []
    for train, test in kfold:
        train_predictors = train_data[predictors].iloc[train, :]
        train_target = train_data["Survived"].iloc[train]

        lin_reg.fit(train_predictors, train_target)
        test_predictions = lin_reg.predict(train_data[predictors].iloc[test, :])

        predictions.append(test_predictions)

    return predictions
