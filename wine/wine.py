import csv
import numpy as np
import lda

# y = true labels
# y_hat = training labels
# return: accuracy of training labels (in percentage)
# Ensure that y and y_hat contain the labels for the same training examples.
def evaluate_acc(y, y_hat):
    score = 0
    for i in range(y.shape[0]):
        if y[i] == y_hat[i]:
            score += 1
    return (score / y.shape[0]) * 100

# y = class labels of training examples
# x = feature data of training examples
# 2 < k = number of folds to use in validation
# return: average of prediction error over the k rounds of execution
def k_fold(x, y, k):
    if k < 1:
        return "Must have at least 1 fold."
    elif k > (x.shape[0]//2):
        return "Too many folds."
    elif k == 1:
        print("1 fold selected - model will be trained and validated on same data set")
        model = lda.LDA(0,0)
        model.fit(x, y)
        return evaluate_acc(y, model.predict(x))
    else:
        rows_per_fold = (x.shape[0] + 1)//k       # a few rows at the end of the training data will be unused
        accuracy = 0

        for exec_round in range(k):
            # determine held-out range
            lower_row = exec_round * rows_per_fold
            upper_row = ((exec_round + 1) * rows_per_fold) - 1
            
            # create validation set
            x_val = np.copy(x)[lower_row:upper_row]
            y_val = np.copy(y)[lower_row:upper_row]

            # create training set
            x_trn = np.concatenate((x[0:lower_row], x[upper_row:]))
            y_trn = np.concatenate((y[0:lower_row], y[upper_row:]))

            # train model
            model = lda.LDA(0,0)
            model.fit(x_trn, y_trn)

            # run validation set through model
            y_hat = model.predict(x_val)
            accuracy += evaluate_acc(y_val, y_hat)

        return accuracy / k


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

# single accuracy test
# wine = lda.LDA(0, 0)                            # create an LDA object
# wine.fit(winedata[:, 0:10], winedata[:, 11])    # call LDA with the training data
# y_hat = wine.predict(winedata[:, 0:10])         # predict the class of all the input data points
# print(evaluate_acc(winedata[:, 11], y_hat))

# k-fold accuracy test
print(k_fold(winedata[:, 0:10], winedata[:, 11], 1))    # evaluate model accuracy using k-fold validation

# compare performance to scikit learn
# from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
# clf = LinearDiscriminantAnalysis()
# clf.fit(winedata[:, 0:10], winedata[:, 11])
# print(evaluate_acc(winedata[:, 11], clf.predict(winedata[:, 0:10])))