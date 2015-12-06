from sklearn.linear_model import LinearRegression
from sklearn.cross_validation import KFold
import preprocess_data as preproc

predictors = ["Pclass", "Age", "Sibsp", "Parch", "Fare", "Embarked"]

lin_reg = LinearRegression()
train_data = preproc.preprocess_silent()

kfold = KFold(train_data.shape[0], n_folds=3, random_state=1)
