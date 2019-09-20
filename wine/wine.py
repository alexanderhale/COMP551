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

lda.lda.fit(winedata[:, 0:10], winedata[:, 11]) # call LDA with the training data