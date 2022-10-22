import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn
import NaiveBayes as nB


def read_data(file_name):
    data = pd.read_csv(file_name)
    return data


def get_data():
    train_data = read_data("dataset/bbcsports_train.csv")
    val_data = read_data("dataset/bbcsports_val.csv")
    X_train = np.array(train_data.iloc[:, :-1])
    y_train = np.array(train_data.iloc[:, -1:])
    X_val = np.array(val_data.iloc[:, :-1])
    y_val = np.array(val_data.iloc[:, -1:])
    return X_train, y_train, X_val, y_val


def get_histogram(class_labels, class_counts):
    plt.bar(class_labels, class_counts)
    plt.show()


def get_confusion_matrix(y_true, y_pred, class_counts):
    y_true = pd.Categorical(y_true, categories=class_counts)
    y_pred = pd.Categorical(y_pred, categories=class_counts)
    confusion_matrix = pd.crosstab(y_true, y_pred, rownames=['Actual'], colnames=['Predicted'], dropna=False)
    print(confusion_matrix)
    sn.heatmap(confusion_matrix, cmap="Blues", annot=True)
    plt.show()


X_train, y_train, X_val, y_val = get_data()
train_labels, train_counts = np.unique(y_train, return_counts=True)
val_labels, val_counts = np.unique(y_val, return_counts=True)
get_histogram(train_labels, train_counts)
get_histogram(val_labels, val_counts)


mleEstimation = nB.NaiveBayes(alpha=0)
prior, likelihood = mleEstimation.fit(X_train, y_train)
y_pred = mleEstimation.predict(prior, likelihood, X_val)
y_true = np.reshape(y_val, (y_val.shape[0]))
accuracyBool = (y_pred == y_true)
print('MlE Accuracy: ', np.count_nonzero(accuracyBool) / accuracyBool.shape[0])
get_confusion_matrix(y_true, y_pred, val_counts)


mapEstimation = nB.NaiveBayes(alpha=1)
prior, likelihood = mapEstimation.fit(X_train, y_train)
y_pred = mapEstimation.predict(prior, likelihood, X_val)
y_true = np.reshape(y_val, (y_val.shape[0]))
accuracyBool = (y_pred == y_true)
print('MAP Accuracy: ', np.count_nonzero(accuracyBool) / accuracyBool.shape[0])
get_confusion_matrix(y_true, y_pred, val_counts)

