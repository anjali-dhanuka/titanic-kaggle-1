from sklearn.ensemble import RandomForestClassifier
from sklearn import cross_validation
import preprocess_data as preproc

predictors = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]


def random_forest_impl(train_data):
    algo = RandomForestClassifier(n_estimators=10, min_samples_split=2,
                                  min_samples_leaf=1, random_state=1)

    score = cross_validation.cross_val_score(algo, train_data[predictors],
                                             train_data["Survived"], cv=3)
    
    print score


if __name__ == '__main__':
    titanic_training_data = preproc.preprocess_silent("train")
    random_forest_impl(titanic_training_data)
