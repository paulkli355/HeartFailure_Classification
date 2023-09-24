import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn import tree
from sklearn.metrics import confusion_matrix

def plot_confusion_matrix(y_pred: np.array, y_test: pd.Series):
    """ Display confusion matrix based on predicted and expected values of target """
    plt.figure(figsize=(3,3))
    cm = confusion_matrix(y_test, y_pred)
    ax = sns.heatmap(cm, annot=True, cmap="Blues")
    ax.set_xlabel('Model Predictions')
    ax.set_ylabel('Actual values')
    plt.title('Confussion Matrix for selected model')
    plt.show()


def plot_model_accuracy_for_epochs(model):
    """ Display model accuracy through per every epoch within learning process """
    plt.figure(figsize=(6,3))
    plt.plot(model.history.history['accuracy'])
    plt.plot(model.history.history['val_accuracy'])
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['train', 'test'])
    plt.show()


def plot_model_loss_for_epochs(model):
    """ Display model loss through per every epoch within learning process """
    plt.figure(figsize=(6,3))
    plt.plot(model.history.history['loss'])
    plt.plot(model.history.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['train', 'test'])
    plt.show()


def plot_tree_based_model(tree_model):
    """ Display graph of tree-based model """
    plt.figure(figsize=(6,3))
    tree.plot_tree(tree_model, filled=True)


def plot_acc_overview(acc_scores: dict):
    """ Display grapgh accuracy comparison between all selected models """
    acc_scores = dict(sorted(acc_scores.items(), key = lambda x: x[1], reverse = True))
    models = list(acc_scores.keys())
    score = list(acc_scores.values())

    fig = plt.figure(figsize=(6, 3))
    sns.barplot(x=score, y=models)
    plt.xlabel("Accuracy Score")
    plt.ylabel("Models Used")
    plt.title("Score for Different Models")
    plt.show()


def plot_mcc_overview(mcc_scores: dict):
    """ Display grapgh Matthews correlation coefficient comparison between all selected models """
    mcc_scores = dict(sorted(mcc_scores.items(), key = lambda x: x[1], reverse = True))
    models = list(mcc_scores.keys())
    score = list(mcc_scores.values())

    fig = plt.figure(figsize=(6, 3))
    sns.barplot(x=score, y=models)
    plt.xlabel("Matthews Correlation Coefficient")
    plt.ylabel("Models Used")
    plt.title("Coefficient Value for Different Models")
    plt.show()


def plot_acc_per_ccp_alpha(ccp_alphas, train_scores, test_scores):
    """ Display accuracy value per CCP Alpha parameter values for training and testing datasets """
    fig, ax = plt.subplots()
    plt.figure(figsize=(4, 2))
    ax.set_xlabel("CCP Alpha")
    ax.set_ylabel("Accuracy")
    ax.set_title("Accuracy value per CCP Alpha parameter value for training and test datasets")
    ax.plot(ccp_alphas, 
            train_scores, 
            marker='*', 
            label='train', 
            drawstyle="steps-post")
    ax.plot(ccp_alphas, 
            test_scores, 
            marker='*', 
            label='test', 
            drawstyle="steps-post")
    ax.legend()
    plt.show()