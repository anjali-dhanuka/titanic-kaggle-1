from sklearn.ensemble import RandomForestClassifier
from sklearn import cross_validation
import preprocess_data as preproc

predictors = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]


def random_forest_impl(train_data):
    algo = RandomForestClassifier(n_estimators=150, min_samples_split=4,
                                  min_samples_leaf=2, random_state=1)

    score = cross_validation.cross_val_score(algo, train_data[predictors],
                                             train_data["Survived"], cv=3)

    print score.mean()


def new_features(train_data):
    train_data["FamilySize"] = train_data["SibSp"] + train_data["Parch"]
    train_data["NameLength"] = train_data["Name"].apply(lambda x: len(x))
    print "Added two new features: FamilySize and NameLength..."

    return train_data


if __name__ == '__main__':
    titanic_training_data = preproc.preprocess_silent("train")
    random_forest_impl(titanic_training_data)

    titanic_training_data = new_features(titanic_training_data)
    predictors.append("FamilySize")
    predictors.append("NameLength")

    random_forest_impl(titanic_training_data)
