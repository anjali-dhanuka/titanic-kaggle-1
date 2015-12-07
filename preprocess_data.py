import pandas as pd
import time


def preprocess_loud(train_or_test):
    print "Preprocessing " + train_or_test + " data: "
    data = pd.read_csv('data/' + train_or_test + '.csv')

    print "Brief description of the data: "
    time.sleep(1)
    print data.describe()

    print "Filling NaN entries of 'Age' with the median of existing entries..."
    data['Age'] = data['Age'].fillna(data['Age'].median())
    time.sleep(1)
    print "Done."

    print "Filling NaN entries of 'Fare' with the median of existing entries..."
    data['Fare'] = data['Fare'].fillna(data['Fare'].median())
    time.sleep(1)
    print "Done."

    print "Manipulating 'Sex' column (male -> 0, female -> 1)..."
    data.loc[data['Sex'] == 'male', 'Sex'] = 0
    data.loc[data['Sex'] == 'female', 'Sex'] = 1
    time.sleep(1)
    print "Done."

    print "Filling NaN entries of 'Embarked' with Southampton('S')..."
    data['Embarked'] = data['Embarked'].fillna("S")
    time.sleep(1)
    print "Done."

    print "Manipulating 'Embarked' column"
    print "(southampton -> 0, cherbourg -> 1, queenstown -> 2)..."
    data.loc[data['Embarked'] == 'S', 'Embarked'] = 0
    data.loc[data['Embarked'] == 'C', 'Embarked'] = 1
    data.loc[data['Embarked'] == 'Q', 'Embarked'] = 2
    time.sleep(1)
    print "Done."

    print "Final description of the data:"
    time.sleep(1)
    print data.describe()


def preprocess_silent(train_or_test):
    data = pd.read_csv('data/' + train_or_test + '.csv')

    data['Age'] = data['Age'].fillna(data['Age'].median())
    data['Fare'] = data['Fare'].fillna(data['Fare'].median())

    data.loc[data['Sex'] == 'male', 'Sex'] = 0
    data.loc[data['Sex'] == 'female', 'Sex'] = 1

    data['Embarked'] = data['Embarked'].fillna('S')
    data.loc[data['Embarked'] == 'S', 'Embarked'] = 0
    data.loc[data['Embarked'] == 'C', 'Embarked'] = 1
    data.loc[data['Embarked'] == 'Q', 'Embarked'] = 2

    return data


if __name__ == '__main__':
    preprocess_loud("train")