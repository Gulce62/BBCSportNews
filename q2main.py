import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn
import NaiveBayes as nB


def read_data(file_name):
    # Read the data with its path location
    try:
        data = pd.read_csv(file_name)
        return data
    except Exception:
        sys.exit(1)


def get_data(file_location):
    # Split the data into X(input) and y(output)
    data = read_data(file_location)
    X = np.array(data.iloc[:, :-1])
    y = np.array(data.iloc[:, -1:])
    return X, y


def get_histogram(class_labels, class_counts):
    # Get the histogram of the data with class labels and number of examples in each label
    plt.figure()
    plt.bar(class_labels, class_counts)


def get_confusion_matrix(y_true, y_pred, class_labels):
    # Get the confusion matrix of the data with the true and predicted labels
    y_true = pd.Categorical(y_true, categories=class_labels)
    y_pred = pd.Categorical(y_pred, categories=class_labels)
    confusion_matrix = pd.crosstab(y_true, y_pred, rownames=['Actual'], colnames=['Predicted'], dropna=False)
    print(confusion_matrix)
    plt.figure()
    sn.heatmap(confusion_matrix, cmap="Blues", annot=True)

def estimate(eName, estimator, prior, likelihood, X, y):
    # Estimate the class labels for validation and test sets with the corresponding estimator (MLE or MAP)
    # Calculate the number of wrong predictions and accuracy
    y_pred = estimator.predict(prior, likelihood, X)
    y_true = np.reshape(y, (y.shape[0]))
    accuracyBool = (y_pred == y_true)
    print('Number of Wrong Prediction for', eName, ': ',  accuracyBool.shape[0]-np.count_nonzero(accuracyBool))
    print(eName, 'Accuracy: ', np.count_nonzero(accuracyBool) / accuracyBool.shape[0])
    return y_pred, y_true

# Get input and read data for train, validation and test sets
train_file_location = input('Enter the file location of the train dataset: ')
X_train, y_train = get_data(train_file_location)
val_file_location = input('Enter the file location of the validation dataset: ')
X_val, y_val = get_data(val_file_location)
test_file_location = input('Enter the file location of the test dataset: ')
X_test, y_test = get_data(test_file_location)

# Get class labels and counts for each label
train_labels, train_counts = np.unique(y_train, return_counts=True)
val_labels, val_counts = np.unique(y_val, return_counts=True)
test_labels, test_counts = np.unique(y_test, return_counts=True)

# Get histogram for train, validation and test sets
get_histogram(train_labels, train_counts)
plt.title('Class Distribution of the Training Set')
get_histogram(val_labels, val_counts)
plt.title('Class Distribution of the Validation Set')
get_histogram(test_labels, test_counts)
plt.title('Class Distribution of the Test Set')

# Fit the training data for both MLE and MAP estimators
mleEstimation = nB.NaiveBayes(alpha=0)
prior_mle, likelihood_mle = mleEstimation.fit(X_train, y_train, train_labels, train_counts)
mapEstimation = nB.NaiveBayes(alpha=1)
prior_map, likelihood_map = mapEstimation.fit(X_train, y_train, train_labels, train_counts)

# Estimate the class labels and get the confusion matrix for validation set for both MLE and MAP estimators
y_pred_val, y_true_val = estimate('Validation Set MLE', mleEstimation, prior_mle, likelihood_mle, X_val, y_val)
get_confusion_matrix(y_true_val, y_pred_val, val_labels)
plt.title('Confusion Matrix of Validation Set with MLE')
y_pred_val, y_true_val = estimate('Validation Set MAP', mapEstimation, prior_map, likelihood_map, X_val, y_val)
get_confusion_matrix(y_true_val, y_pred_val, val_labels)
plt.title('Confusion Matrix of Validation Set with MAP')

# Estimate the class labels and get the confusion matrix for test set for both MLE and MAP estimators
y_pred_test, y_true_test = estimate('Validation Set MLE', mleEstimation, prior_mle, likelihood_mle, X_test, y_test)
get_confusion_matrix(y_true_test, y_pred_test, test_labels)
plt.title('Confusion Matrix of Test Set with MLE')
y_pred_test, y_true_test = estimate('Validation Set MAP', mapEstimation, prior_map, likelihood_map, X_test, y_test)
get_confusion_matrix(y_true_test, y_pred_test, test_labels)
plt.title('Confusion Matrix of Test Set with MAP')

# Show all the plots
plt.show()
