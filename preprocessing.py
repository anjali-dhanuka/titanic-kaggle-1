import pandas as pd
import time


def preprocess():
    train_data = pd.read_csv('data/train.csv')

    print "Brief description of the data: "
    time.sleep(1)
    print train_data.describe()

    print "Filling NaN entries of 'Age' with the median of existing entries..."
    train_data['Age'] = train_data['Age'].fillna(train_data['Age'].median())
    time.sleep(1)
    print "Done."

    print "Manipulating 'Sex' column (male -> 0, female -> 1)..."
    train_data.loc[train_data['Sex'] == 'male', 'Sex'] = 0
    train_data.loc[train_data['Sex'] == 'female', 'Sex'] = 1
    time.sleep(1)
    print "Done."

    print "Manipulating 'Embarked' column"
    print "(southampton -> 0, cherbourg -> 1, queenstown -> 2)..."
    train_data.loc[train_data['Embarked'] == 'S', 'Embarked'] = 0
    train_data.loc[train_data['Embarked'] == 'C', 'Embarked'] = 1
    train_data.loc[train_data['Embarked'] == 'Q', 'Embarked'] = 2
    time.sleep(1)
    print "Done."

    print "Final description of the data:"
    time.sleep(1)
    print train_data.describe()


if __name__ == '__main__':
    preprocess()