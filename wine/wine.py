import csv
import numpy as np
import lda

# column titles:
    # "fixed acidity";
    # "volatile acidity";
    # "citric acid";
    # "residual sugar";
    # "chlorides";
    # "free sulfur dioxide";
    # "total sulfur dioxide";
    # "density";
    # "pH";
    # "sulphates";
    # "alcohol";
    # "quality"

# y = true labels
# y_hat = training labels
# return: accuracy of training labels (in percentage)
# Ensure that y and y_hat contain the labels for the same training examples.
def evaluate_acc (x, y, y_hat):
    # TODO input x is not really needed to evaluate accuracy, but project instructions specify it
    score = 0
    for i in range(y.shape[0]):
        if y[i] == y_hat[i]:
            score += 1
    return (score / y.shape[0]) * 100

# import data from CSV file
with open('winequality-red.csv', 'r') as inputfile:
    winedata = list(csv.reader(inputfile, delimiter=';'))

# convert data to NumPy 2D array
winedata = np.array(winedata[1:], dtype=np.float)

# turn "quality" attribute to binary
for row in winedata:
    if row[11] >= 6:
        row[11] = 1
    else:
        row[11] = 0

wine = lda.LDA(0, 0)                            # create an LDA object
wine.fit(winedata[:, 0:10], winedata[:, 11])    # call LDA with the training data
y_hat = wine.predict(winedata[:, 0:10])         # predict the class of all the input data points
print(evaluate_acc(winedata[:, 0:10], winedata[:, 11], y_hat))  # find out the accuracy of the predictions