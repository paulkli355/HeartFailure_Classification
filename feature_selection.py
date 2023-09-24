import pandas as pd

from sklearn.feature_selection import VarianceThreshold, mutual_info_classif, SelectKBest, chi2

def chi2_feature_selection(x_train: pd.DataFrame, y_train: pd.Series):
    assert x_train.isnull().sum().sum() == 0
    f_p_values = chi2(x_train, y_train)
    p_values = pd.Series(f_p_values[1])
    p_values.index = x_train.columns

    # P values sorted ascending will show at the top most important features
    p_values.sort_values(ascending=True, inplace=True)

    return p_values

def exclude_low_variance_features(x_train: pd.DataFrame, x_test: pd.DataFrame, threshold: float):
    """ Exclude features with 0 or lower than given threshold variance from training and testing dataset """
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


