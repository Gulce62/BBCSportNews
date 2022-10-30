# CS 464 Homework 1 - BBCSportNews Classification

Required programming language: Python 

Required libraries: numpy, pandas, matplotlib and seaborn (sys library is used to exit if the file path is incorrect)

The program consists of the main file (q2main.py) and the file that includes Naive Bayes class (NaiveBayes.py).
In the q2main.py file the csv data is read and converted into input and output arrays. The data distribution graphs are plotted.
Then the Naive Bayes class is called for both MLE and MAP estimators. Both for validation and test data the desired 
calculations and plots are displayed. In NaiveBayes.py file, there is Naive Bayes class that includes a fit and predict function.
The fit function calculates likelihoods and priors and the predict function estimates the class label for validation and test data. 

How to execute: If the libraries are not installed used the command 'pip install x' in the command window.
From terminal/command window run q2main.py with the command 'python q2main.py'. It is needed to input the file 
locations/directories of the training, validation and test data respectively. If the directory is wrong, the program ends.
The Dirichlet prior is taken 0 for MLE and 1 for MAP estimator as default.

Outputs: The program will display the number of wrong predictions, the accuracy and the confusion matrix for both 
validation and test sets in terminal. The program will output the distribution graphs and confusion matrix plots for 
training, validation and test data. 

To rerun the program, it is needed to close all the plot figures. 

