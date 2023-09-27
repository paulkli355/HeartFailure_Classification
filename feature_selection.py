import pandas as pd

from sklearn.feature_selection import VarianceThreshold, mutual_info_classif, chi2
from sklearn.feature_selection import SelectKBest, RFE, SelectFromModel
from sklearn.ensemble import RandomForestClassifier

def chi2_feature_selection(train_fold: pd.DataFrame, target_col: str):
    x_train = train_fold.drop(columns=[target_col])
    y_train = train_fold[target_col]

    assert x_train.isnull().sum().sum() == 0
    f_p_values = chi2(x_train, y_train)
    p_values = pd.Series(f_p_values[1])
    p_values.index = x_train.columns

    # P values sorted ascending will show at the top most important features
    p_values.sort_values(ascending=True, inplace=True)

    return p_values

def exclude_low_variance_features(train_fold: pd.DataFrame, test_fold: pd.DataFrame, target_col: str, threshold: float):
    """ Exclude features with 0 or lower than given threshold variance from training and testing dataset """
    x_train = train_fold.drop(columns=[target_col])
    y_train = train_fold[target_col]
    x_test = test_fold.drop(columns=[target_col])
    y_test = test_fold[target_col]

    var_thr = VarianceThreshold(threshold)
    var_thr.fit(x_train)

    const_cols = [column for column in x_train.columns if column not in x_train.columns[var_thr.get_support()]]

    if const_cols:
        x_train.drop(columns=const_cols, inplace=True)
        x_test.drop(columns=const_cols, inplace=True)

    return x_train, x_test

def exclude_correlated_features(x_train: pd.DataFrame, x_test: pd.DataFrame, corr_matrix: pd.DataFrame, threshold: float):
    """ Check which features are highly correlated with eachother 
        and exclude redundant ones from training and testing datasets
    """
    col_corr = set()
    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            if corr_matrix.iloc[i, j] > threshold: # abs(corr_matrix.iloc[i,j]) > threshold:
                col_name = corr_matrix.columns[i]
                col_corr.add(col_name)

    if col_corr:
        x_train.drop(columns=col_corr, inplace=True)
        x_test.drop(columns=col_corr, inplace=True)

    return x_train, x_test

def select_features_using_information_gain(x_train: pd.DataFrame, y_train: pd.Series):
    mutual_info = pd.Series(mutual_info_classif(x_train, y_train))
    mutual_info.index = x_train.columns
    mutual_info.sort_values(ascending=False, inplace=True)

    selected_top_features = SelectKBest(mutual_info, k=10)
    selected_top_features.fit_transform(x_train.fillna(0), y_train)
    selected_top_cols = x_train.columns[selected_top_features.get_support()]

    return x_train[[selected_top_cols]]


