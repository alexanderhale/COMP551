import numpy as np

class LogisticRegression:
    def __init__(self, alpha=0.001, threshold = 0.0005):
        self.alpha = alpha
        self.threshold = threshold
        self.stop = False
        self.weights = None
        self.max_iter = 10000
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
        self.change = [] # reset the gradients before running a new fit
        padded_X = self.__intercept(X)
        self.weights = np.zeros(np.size(padded_X,1))
        
        num_iter = 0
        while self.change == [] or self.change[-1] > self.threshold and num_iter < self.max_iter:
            self.__update(padded_X, Y)
            num_iter+=1
            
            if (num_iter == self.max_iter):
                print(f"Warning, reached max iterations of {self.max_iter}, stopping because we haven't converged yet")
                break

        print(f"learning rate:{self.alpha} \n stop threshold:{self.threshold} \n number of iterations: {num_iter}")
        print(f"weights:{self.weights}")
        
        return self.weights
    
    def predict(self, X):
        padded_X = self.__intercept(X)
        predictions = []
        
        for i in range(0, len(X)):
            Z = np.dot(self.weights.T, padded_X[i])
            pred = self.__sigmoid(Z).round()
            predictions.append(pred)
        
        return predictions