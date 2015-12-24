from sklearn.ensemble import RandomForestClassifier
from sklearn import cross_validation
import preprocess_data as preproc
import re

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

    titles = train_data["Name"].apply(get_title)

    title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Dr": 5, "Rev": 6, "Major": 7, "Col": 7,
                     "Mlle": 8, "Mme": 8, "Don": 9, "Lady": 10, "Countess": 10, "Jonkheer": 10, "Sir": 9,
                     "Capt": 7, "Ms": 2}

    for key, value in title_mapping.items():
        titles[titles == key] = value

    train_data["Title"] = titles
    print "Added a new feature: Title..."

    return train_data


def get_title(name):
    title_search = re.search(" ([A-Za-z]+)\.", name)

    if title_search:
        return title_search.group(1)
    return ""


if __name__ == '__main__':
    titanic_training_data = preproc.preprocess_silent("train")
    random_forest_impl(titanic_training_data)

    titanic_training_data = new_features(titanic_training_data)
    predictors.append("FamilySize")
    predictors.append("NameLength")
    predictors.append("Title")
    random_forest_impl(titanic_training_data)
