import pandas as pd

from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from scipy import stats
from tensorflow.keras.utils import to_categorical


def encode_categorical_features(dataset: pd.DataFrame, target_col: str):
    """ Check Data Types using the dtypes attribute of a DataFrame. 
        Categorical variables are typically of type object or category.
        For categorical columns, apply the appropriate encoding technique (label encoding or one-hot encoding)
    """
    ordinal_f, nominal_f = detect_ordinal_and_nominal_features(dataset=dataset, target_col=target_col)

    # Apply label encoding to ordinal categorical variables
    label_encoder = LabelEncoder()
    for feature_col in ordinal_f:
        dataset[feature_col] = dataset[feature_col].apply(label_encoder.fit_transform)

    # Apply one-hot encoding to nominal categorical variables
    one_hot_encoder = OneHotEncoder(sparse=False, drop='first')
    for feature_col in nominal_f:
        encoded_features = one_hot_encoder.fit_transform(dataset[[feature_col]])
        data_encoded = pd.DataFrame(encoded_features, columns=one_hot_encoder.get_feature_names([feature_col]))

        # replace old values with encoded ones
        # dataset[feature_col] = data_encoded
        # Concatenate the encoded features with the original dataset and drop the original feature column
        dataset = pd.concat([dataset, data_encoded], axis=1)
        dataset = dataset.drop(columns=[feature_col])

    return dataset



def detect_ordinal_and_nominal_features(dataset: pd.DataFrame, target_col: str):
    """ Search for categorical features in dataset and divide them into two lists: ordinal and nominal ones """
    # Identify categorical columns
    categorical_columns = dataset.select_dtypes(include=['object', 'category']).columns

    # Create empty lists to store ordinal and nominal features
    ordinal_categorical_features = []
    nominal_categorical_features = []

    # Loop through categorical columns and use statistical tests to classify them
    for col in categorical_columns:
        unique_values = dataset[col].nunique()
        
        # Perform a Chi-Square test to check for association with the target variable
        chi2_stat, p_val, dof, _ = dataset.chi2_contingency(pd.crosstab(dataset[col], dataset[target_col]))
        
        if unique_values <= 10:
            # If the number of unique values is small, consider it ordinal
            ordinal_categorical_features.append(col)
        elif p_val < 0.05:
            # If p-value is less than 0.05, consider it nominal (significant association)
            nominal_categorical_features.append(col)

    print("Ordinal Categorical Features:", ordinal_categorical_features)
    print("Nominal Categorical Features:", nominal_categorical_features)

    return ordinal_categorical_features, nominal_categorical_features


def rescale_features(dataset: pd.DataFrame):
    """ Transform values that are in different scales into rescaled format, without information loss """
    # some models are sensitive for feature value scale differences
    standard_scaler = StandardScaler()
    rescaled_dataset = pd.DataFrame(standard_scaler.fit_transform(dataset))

    return rescaled_dataset


def transform_target_to_categorical(target_df: pd.DataFrame):
    """ Transform target data sets into binary matrix representation of the target values """
    categorical_target = to_categorical(target_df)

    return categorical_target
