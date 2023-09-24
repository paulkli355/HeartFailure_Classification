import pandas as pd
import numpy as np

from visualization import plot_confusion_matrix
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef, roc_curve, auc

def calculate_model_predictions(model, x_test: pd.Series) -> np.array:
    """ Calculate predictions for selected model
        based on set of features' values from testing dataset (x_test)
    """
    model_predictions = model.predict(x_test)

    return model_predictions


def calculate_metrics(y_pred: np.array, y_test: pd.Series):
    """ Calculate model quality metrics based on 
        expected label values from testing dataset (y_test) and predicted values.
    """
    model_acc = accuracy_score(y_test, y_pred)
    model_precision = precision_score(y_test, y_pred)
    model_recall = recall_score(y_test, y_pred)
    model_f1_score = f1_score(y_test, y_pred)
    model_classification_report = classification_report(y_test, y_pred)
    model_mcc = matthews_corrcoef(y_test, y_pred)

    print(f'Model accuracy: {model_acc*100}%')
    print(f'Model precision: {model_precision*100}%')
    print(f'Model recall: {model_recall*100}%')
    print(f'Model F1-score: {model_f1_score}')
    print(f'Model Matthews Correlation Coefficient (MCC): {model_mcc}')
    print(f'Model overall classification report:\n \n{model_classification_report}')

    plot_confusion_matrix(y_pred=y_pred, y_test=y_test)

    return model_acc, model_mcc


def evaluate_model(trained_model, x_test: pd.Series, y_test: pd.Series, is_cnn:bool = False):
    """ Based on trained model, proceed with classification and calculate predictions.
        Then calculate model accuracy metrics based on expected values from testing dataset.
    """
    model_predictions = calculate_model_predictions(model=trained_model,
                                                    x_test=x_test)
    if is_cnn:
        model_predictions = np.argmax(model_predictions, axis=1)
    
    acc, mcc = calculate_metrics(y_pred=model_predictions, 
                      y_test=y_test)
    
    return acc, mcc

