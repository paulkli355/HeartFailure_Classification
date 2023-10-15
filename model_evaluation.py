import pandas as pd
import numpy as np
import os

from math import sqrt
from visualization import plot_confusion_matrix, plot_roc_curve
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_score, recall_score, f1_score, \
    matthews_corrcoef, mean_squared_error, r2_score, roc_auc_score, roc_curve, auc


def calculate_model_predictions(model, x_test: pd.DataFrame) -> np.array:
    """ Calculate predictions for selected model
        based on set of features' values from testing dataset (x_test)
    """
    model_predictions = model.predict(x_test)

    return model_predictions


def calculate_metrics(y_pred: np.array, y_test: pd.Series):
    """ Calculate model quality metrics based on 
        expected label values from testing dataset (y_test) and predicted values.
    """
    tn, fp, fn, tp = calculate_test_results_from_confusion_matrix(y_test, y_pred)
    model_precision = precision_score(y_test, y_pred)
    model_recall = recall_score(y_test, y_pred) # sensitivity
    model_specificity = specificity_score(tn, fp)
    model_acc = accuracy_score(y_test, y_pred)
    model_npv = calculate_npv(tn, fn)
    
    model_f1_score = f1_score(y_test, y_pred)
    model_mcc = matthews_corrcoef(y_test, y_pred)
    rmse = sqrt(mean_squared_error(y_test, y_pred))
    model_r2 = r2_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred)
    fpr, tpr, thresholds = roc_curve(y_test, y_pred)
    model_classification_report = classification_report(y_test, y_pred)

    model_scores = {
        'True Negative': tn,
        'False Positive': fp,
        'False Negative': fn,
        'True Positive': tp,
        'Precision (PPV)': model_precision,
        'Sensitivity (TPR, Recall)': model_recall,
        'Speciticity (TNR)': model_specificity,
        'Accuracy': model_acc,
        'Negative Predictive Value (NPV)': model_npv,
        'F1 Score': model_f1_score,
        'RMSE': rmse,
        'R Squared': model_r2,
        'Matthews Correlation Coefficient (MCC)': model_mcc,
        'Threshold (from ROC Curve)': thresholds,
        'False Positive Rate (FPR)': fpr,
        'ROC AUC score': roc_auc,
    }

    print(f'Base error test results: TN {tn}, FP {fp}, FN {fn}, TP {tp}')
    print(f'Precision (PPV): {round(model_precision*100,2)}%')
    print(f'Recall (Sensitivity, TPR): {round(model_recall*100,2)}%')
    print(f'Specificity (TNR): {round(model_specificity*100,2)}%')

    print(f'Accuracy: {round(model_acc*100,2)}%')
    print(f'F1-score: {round(model_f1_score,2)}')
    print(f'Matthews Correlation Coefficient (MCC): {round(model_mcc,2)}')
    print(f'RMSE: {round(rmse,2)}')
    print(f'R Squared: {round(model_r2,2)}')
    print(f'ROC AUC Score: {roc_auc}')
    print(f'Model overall classification report:\n \n{model_classification_report}')

    plot_confusion_matrix(y_pred=y_pred, y_test=y_test)
    plot_roc_curve(y_pred=y_pred, y_test=y_test)

    return model_acc, model_mcc


def evaluate_model(trained_model, x_test: pd.DataFrame, y_test: pd.Series, is_cnn:bool = False):
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


def calculate_test_results_from_confusion_matrix(y_test: pd.DataFrame, y_pred: pd.DataFrame):
    """ Calculate the confusion matrix and extract TP, FP, TN, FN from that matrix """
    conf_matrix = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = conf_matrix.ravel()

    return tn, fp, fn, tp


def specificity_score(tn: float, fp: float):
    return tn / (tn + fp)


def calculate_npv(tn: float, fn: float):
    return tn / (tn + fn)

