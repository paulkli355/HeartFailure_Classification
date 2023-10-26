import os
import openpyxl
import pandas as pd

from sklearn.svm import LinearSVC
from sklearn.feature_selection import chi2, mutual_info_classif, VarianceThreshold, SelectFromModel


def remove_low_variance_features(X_train: pd.DataFrame) -> tuple:
    vthresh = VarianceThreshold(threshold=0.005)
    vthresh.fit_transform(X_train)
    selected_features_vth = vthresh.get_feature_names_out()
    # print(f'Variance Threshold (0.005) selected features are: {selected_features_vth}')

    return set(selected_features_vth)


def select_importatnt_features_by_chi2(X_train: pd.DataFrame, y_train: pd.DataFrame) -> tuple:
    alpha = 0.05

    f_score, p_values = chi2(X_train, y_train)
    p_values = pd.Series(p_values)
    p_values.index = X_train.columns
    p_values.sort_values(ascending=False, inplace=True)
    selected_features_chi2 = list(p_values[p_values < alpha].index)
    # print(f'Chi2 test selected features are: {selected_features_chi2}')

    return set(selected_features_chi2)


def select_importatnt_features_by_mi(X_train: pd.DataFrame, y_train: pd.DataFrame) -> tuple:
    threshold = 0.01

    importances = mutual_info_classif(X_train,y_train)
    feat_importances = pd.Series(importances, X_train.columns).sort_values()
    threshold = 0.01
    selected_features_mi = list(feat_importances[feat_importances > threshold].index)
    # print(f'Mutual infromation estimation selected features are: {selected_features_mi}')

    return set(selected_features_mi)


def select_importatnt_features_by_corr(X_train: pd.DataFrame) -> tuple:
    """ This function removes features that are highly correlated with each other. 
        It doesn't check the target-feature correlation.
    """
    def correlation(dataset, threshold):
        col_corr = set()  # Set of all the names of correlated columns
        corr_matrix = dataset.corr()
        for i in range(len(corr_matrix.columns)):
            for j in range(i):
                if abs(corr_matrix.iloc[i, j]) > threshold: # we are interested in absolute coeff value
                    colname = corr_matrix.columns[i]  # getting the name of column
                    col_corr.add(colname)
        return col_corr
    
    high_corr_features = set(correlation(X_train, 0.8))
    selected_features_corr = [item for item in X_train.columns if item not in high_corr_features]
    # print(f'Pearson correlation selected features are: {selected_features_corr}')

    return set(selected_features_corr)


def select_importatnt_features_by_l1_reg(X_train: pd.DataFrame, y_train: pd.DataFrame) -> tuple:
    lsvc = LinearSVC(C=0.0001, penalty="l1", dual=False, max_iter=1500).fit(X_train, y_train)
    sfm = SelectFromModel(lsvc, prefit=True)

    X_train_new = X_train.loc[:, sfm.get_support()]
    selected_features_l1 = X_train_new.columns
    # print(f'Lasso L1 Regularization selected features are: {selected_features_l1}')

    return set(selected_features_l1)


def select_common_features(X_train: pd.DataFrame, y_train: pd.DataFrame, fold_num: int = 0) -> list:
    vrh_sf = remove_low_variance_features(X_train)
    chi2_sf = select_importatnt_features_by_chi2(X_train, y_train)
    mi_sf = select_importatnt_features_by_mi(X_train, y_train)
    corr_sf = select_importatnt_features_by_corr(X_train)
    l1_sf = select_importatnt_features_by_l1_reg(X_train, y_train)

    methods = ['excl_low_variance', 'chi2', 'mutual_info', 'corr', 'l1_reg']
    features_matrix = pd.DataFrame(columns=X_train.columns, index=methods)

    for column in X_train.columns:
        for method_name, col_set in zip(methods, [vrh_sf, chi2_sf, mi_sf, corr_sf, l1_sf]):
            is_important_col = column in col_set
            features_matrix.at[method_name, column] = is_important_col
    
    # Create a directory if it doesn't exist
    if not os.path.exists('feature_selection_process'):
        os.makedirs('feature_selection_process')

    features_matrix.to_excel(f'feature_selection_process/selected_features_fold_no{fold_num}.xlsx')

    common_features = list(vrh_sf.intersection(chi2_sf, mi_sf, corr_sf, l1_sf))
    # print(f'Across all methods there were {len(common_features)} selected. These are: {common_features}')

    return set(common_features)