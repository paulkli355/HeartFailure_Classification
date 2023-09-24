import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

from catboost import CatBoostClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

from model_evaluation import evaluate_model


def decision_tree_classifier(x_train: pd.Series, y_train: pd.Series, x_test: pd.Series, y_test: pd.Series):
    """ Create and train Decision Tree classifier """
    dt_model = DecisionTreeClassifier(random_state=11, 
                                    criterion='entropy')
    dt_model.fit(x_train, y_train)
    dt_acc, dt_mcc = evaluate_model(dt_model, x_test=x_test, y_test=y_test)