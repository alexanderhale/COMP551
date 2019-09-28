import numpy as np

class LogisticRegression:
    def __init__(self, alpha=0.001, threshold = 0.00005):
        self.alpha = alpha
        self.threshold = threshold
        self.stop = False
        self.weights = None
        self.change = []

    def __intercept(self, X):
        return np.c_[np.ones(len(X)), X]
    
    def __sigmoid(self, z):
        return 1/(1 + np.exp(-z))
    
    def __grad(self, X_i, y_i):
        z = np.dot(self.weights.T, X_i)
        return X_i*(y_i-self.__sigmoid(z))
    
    def __update(self, X, Y):
        changeW = np.zeros(np.size(X, 1))

        for i in range(len(X)):
            grad = self.__grad(X[i], Y[i])
            changeW = changeW + self.alpha*grad
        self.change.append(np.linalg.norm(changeW))
        self.weights = self.weights + changeW
    
    def fit(self, X, Y):
        padded_X = self.__intercept(X)
        self.weights = np.zeros(np.size(padded_X,1))
        
        num_iter = 0
        while self.change == [] or self.change[-1] > self.threshold:
            self.__update(padded_X, Y)
            num_iter+=1

        print(f"learning rate:{self.alpha} \n stop threshold:{self.threshold} \n number of iterations: {num_iter}")
        return self.weights
    
    def predict(self, X):
        padded_X = self.__intercept(X)
        return self.__sigmoid(np.dot(self.weights.T, padded_X)).round()