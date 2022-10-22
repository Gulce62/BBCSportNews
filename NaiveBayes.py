import numpy as np


class NaiveBayes:
    def __init__(self, alpha):
        self.alpha = alpha

    def fit(self, X_train, y_train):
        class_labels, class_counts = np.unique(y_train, return_counts=True)
        prior = class_counts / y_train.shape[0]
        likelihood = []
        for c in class_labels:
            indices = np.where(y_train == c)[0]
            word_occurrences = np.sum(X_train[indices], axis=0) + self.alpha
            total_occurrences = np.sum(X_train[indices]) + (self.alpha * X_train.shape[1])
            likelihood.append(word_occurrences / total_occurrences)
        return prior, np.array(likelihood)

    def predict(self, prior, likelihood, X):
        frequencies = np.reshape(X, (X.shape[0], 1, X.shape[1]))
        log_likelihood = np.log(likelihood)
        log_prior = np.log(prior)
        mult_freq = np.multiply(log_likelihood, frequencies)
        summation = np.sum(mult_freq, axis=2)
        summation += np.resize(log_prior, summation.shape)
        max_indices = np.argmax(summation, axis=1)
        return max_indices
