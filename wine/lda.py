import numpy as np

# Linear Discriminant Analysis
class LDA:
    def __init__(self, w0, w):
        self.w0 = w0     # weight parameter 0
        self.w = w       # initialized as zero, but later stores a vector of other weights

    # x = 2D array containing training data
    # y = results of that training data
    # doesn't return anything, but finishes by modifying model parameters w0 and w
    def fit(self, x, y):
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
        # TODO calculate this with the formula in the slides instead of the built-in NumPy covariance function
        sigma = np.cov(x, rowvar=False)              # stores the covariance matrix of x
        sigma_inverse = np.linalg.inv(sigma)

        # calculate log-odds result
        self.w0 = np.log(np.true_divide(p1, p0))

        part2 = np.transpose(mu1)
        part2 = np.dot(part2, sigma_inverse)
        part2 = np.dot(part2, mu1)
        self.w0 -= np.true_divide(part2, 2)

        part3 = np.transpose(mu0)
        part3 = np.dot(part3, sigma_inverse)
        part3 = np.dot(part3, mu0)
        self.w0 += np.true_divide(part3, 2)              # w0 stores weight parameter w_0

        self.w = np.subtract(mu1, mu0)
        self.w = np.dot(sigma_inverse, self.w)           # w stores the vector of other weight parameters

    # x = 2D array containing input points upon which predictions will be made
    # return: vector y^, containing binary predictions about whether each input point is class 0 or 1
        # y^[0] is prediction for x's first input point row, y^[1] the second input point, etc
    def predict(self, x):
        y = np.zeros(x.shape[0])
        for i in range(y.shape[0]):
            # find the probability of each data point being in class 1:
            #   multiply each input data point by the stored weight parameter, and add in bias term
                # TODO figure out why this probability is not landing between 0 and 1
            if self.w0 + np.dot(x[i].T, self.w) > 0.5:
                y[i] = 1        # if probability > 0.5, predict class 1
        return y
