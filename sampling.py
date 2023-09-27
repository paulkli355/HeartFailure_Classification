import os
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold

def create_stratified_kfolds(x_df: pd.DataFrame, y_df: pd.DataFrame, dataset: pd.DataFrame, n_splits: int):
    """ Proceed with stratified sampling of a dataset into given number of folds 
        and save them independently into separate CSV files 
    """
    # Create a directory for the folds if it doesn't exist
    if not os.path.exists('folds'):
        os.makedirs('folds')

    stratified_kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    # Create and save the folds
    for fold_num, (train_index, test_index) in enumerate(stratified_kfold.split(x_df, y_df), 1):
        # Create fold-specific DataFrames
        train_fold = dataset.iloc[train_index]
        test_fold = dataset.iloc[test_index]
        
        # Save each fold as a CSV file
        train_fold.to_csv(f'folds/fold_{fold_num}_train.csv', index=False)
        test_fold.to_csv(f'folds/fold_{fold_num}_test.csv', index=False)

    print(f'{n_splits} Folds created and saved in the "folds" directory successfully')


def create_train_test_sets(x_df: pd.DataFrame, y_df: pd.DataFrame, test_size: float):
    """ Create train and test sets from given dataset. 
        Split will happen accordingly to provided test_size parameter value (%)
    """
    # Select training and test datasets
    x_train, x_test, y_train, y_test = train_test_split(x_df, 
                                                        y_df, 
                                                        random_state=12, 
                                                        test_size=test_size)

    print(f"Training dataset consists of {x_train.shape[0]} records")
    print(f"Test dataset consists of {x_test.shape[0]} records")

    return x_train, x_test, y_train, y_test