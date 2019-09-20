import numpy as np

# Linear Discriminant Analysis
class lda:
    # x = 2D array containing training data
    # y = results of that training data
    def fit(x, y):
        # def __init__():

        # calculate P(y = 0) and P(y = 1)
        n0 = 0                              # stores number of results in class 0
        for datapoint in y:
            if datapoint == 0:
                n0 += 1
        p0 = n0 / y.size                    # stores probability of result being class 0
        p1 = 1 - p0                         # stores probability of result being in class 1

        # calculate mu0 and mu1
        mu0 = np.zeros(x[0].size)
        mu1 = np.zeros(x[0].size)
        for i in range(y.size):
            if y[i] == 0:
                mu0 = np.add(mu0, x[i][:])
            else:
                mu1 = np.add(mu1, x[i][:])                        
        mu0 = np.true_divide(mu0, n0)               # stores matrix of mean of features when result is class 0
        mu1 = np.true_divide(mu1, (y.size - n0))    # stores matrix of mean of features when result is class 1

        # calculate sigma
        sigma = np.cov(x)                       # stores the covariance matrix of x (# TODO should this be done manually, or is using np.cov ok?)
        sigma_inverse = np.linalg.inv(sigma)    # TODO fix the singular matrix error that is coming up here because sigma is a square matrix

        # calculate log-odds result
        part1 = np.log(np.true_divide(p1, p0))

        part2 = np.transpose(mu1)
        part2 = np.dot(part2, sigma_inverse)
        part2 = np.dot(part2, mu1)
        part2 = np.true_divide(part2, 2)

        part3 = np.transpose(mu0)
        part3 = np.dot(part3, sigma_inverse)
        part3 = np.dot(part3, mu0)
        part3 = np.true_divide(part3, 2)

        part4 = np.transpose(x)
        part4 = np.dot(part4, sigma_inverse)
        part4_1 = np.subtract(mu1, mu0)
        part4 = np.dot(part4, part4_1)

        result = np.subtract(part1, part2)
        result = np.add(result, part3)
        result = np.add(result, part4)

        print(result)
    # def predict():