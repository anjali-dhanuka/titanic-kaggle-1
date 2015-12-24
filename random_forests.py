from sklearn.ensemble import RandomForestClassifier
from sklearn import cross_validation
import preprocess_data as preproc
import re

predictors = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]


def random_forest_impl(titanic):
    algo = RandomForestClassifier(n_estimators=200, min_samples_split=4,
                                  min_samples_leaf=2, random_state=1)

    score = cross_validation.cross_val_score(algo, titanic[predictors],
                                             titanic["Survived"], cv=3)

    return score.mean()


def add_feat_family_size(titanic):
    titanic["FamilySize"] = titanic["SibSp"] + titanic["Parch"]
    predictors.append("FamilySize")

    return titanic


def add_feat_name_length(titanic):
    titanic["NameLength"] = titanic["Name"].apply(lambda x:len(x))
    predictors.append("NameLength")

    return titanic


def add_feat_title(titanic):
    titles = titanic["Name"].apply(get_title)

    title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Dr": 5, "Rev": 6, "Major": 7, "Col": 7,
                     "Mlle": 8, "Mme": 8, "Don": 9, "Lady": 10, "Countess": 10, "Jonkheer": 10, "Sir": 9,
                     "Capt": 7, "Ms": 2}

    for key, value in title_mapping.items():
        titles[titles == key] = value
    titanic["Title"] = titles
    predictors.append("Title")
    return titanic


def get_title(name):
    title_search = re.search(" ([A-Za-z]+)\.", name)

    if title_search:
        return title_search.group(1)
    return ""


if __name__ == '__main__':
    titanic_train = preproc.preprocess_silent("train")
    cross_val_score = random_forest_impl(titanic_train)
    print "Cross Validation Score: " + str(cross_val_score)

    titanic_train = add_feat_name_length(titanic_train)
    print "Added new feature: NameLength."
    cross_val_score = random_forest_impl(titanic_train)
    print "Improved Cross Validation Score: " + str(cross_val_score)

    titanic_train = add_feat_family_size(titanic_train)
    print "Added new feature: FamilySize."
    cross_val_score = random_forest_impl(titanic_train)
    print "Improved Cross Validation Score: " + str(cross_val_score)

    titanic_train = add_feat_title(titanic_train)
    print "Added new feature: Title."
    cross_val_score = random_forest_impl(titanic_train)
    print "Improved Cross Validation Score: " + str(cross_val_score)
