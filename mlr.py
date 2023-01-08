import numpy as np


class MultinomialLogisticRegression:
    def __init__(self, C=1, penalty='l2', learning_rate=0.01, epochs=1000, batch_size=32):
        self.C = C
        self.penalty = penalty
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.W = None
        self.b = None

    def get_params(self, deep=True):
        return {"C": self.C, "penalty": self.penalty, "learning_rate": self.learning_rate, "epochs": self.epochs, "batch_size": self.batch_size}

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self

    def softmax(self, z):
        return np.exp(z) / np.sum(np.exp(z), axis=1, keepdims=True)

    def cross_entropy(self, Y, Y_hat):
        loss = -np.sum(Y * np.log(Y_hat)) / Y.shape[0]
        if self.penalty == 'l2':
            loss += 1 / (2 * Y.shape[0]) * self.C * np.sum(np.square(self.W))
        elif self.penalty == 'l1':
            loss += 1 / (2 * Y.shape[0]) * self.C * np.sum(np.abs(self.W))
        return loss

    def fit(self, X, Y):
        self.W = np.zeros((X.shape[1], Y.shape[1]))
        self.b = np.zeros((1, Y.shape[1]))

#         print(f'W Shape: {self.W.shape}, b Shape: {self.b.shape}')

        for epoch in range(self.epochs):
            num_batches = X.shape[0] // self.batch_size

            for batch in range(num_batches):
                X_batch = X[batch*self.batch_size: (batch+1)*self.batch_size]
                Y_batch = Y[batch*self.batch_size: (batch+1)*self.batch_size]

                z = np.dot(X_batch, self.W) + self.b
                Y_hat = self.softmax(z)
                loss = self.cross_entropy(Y_batch, Y_hat)
                if (epoch + 1) % 25 == 0:
                    print(f'Batch: {batch}, Epoch: {epoch}, Loss: {loss}')

                dW = np.dot(X_batch.T, (Y_hat - Y_batch)) / num_batches
                db = np.sum(Y_hat - Y_batch, axis=0) / num_batches

                self.W -= self.learning_rate * dW
                self.b -= self.learning_rate * db

#                 print(f'W: {self.W}, bias: {self.b}')

    def predict(self, X):
        z = np.dot(X, self.W) + self.b
        Y_hat = self.softmax(z)
        return np.argmax(Y_hat, axis=1)

    def score(self, X, Y):
        Y_hat = self.predict(X)
        Y = np.argmax(Y, axis=1)
        return (np.sum(Y_hat == Y)).astype(np.float64) / np.array(Y.shape[0]).astype(np.float64)

    def onehot(self, Y):
        # Constructing zero matrix
        Y_onehot = np.zeros((Y.shape[0], Y.max() + 1))

        # Giving 1 for some colums
        Y_onehot[np.arange(Y.shape[0]), Y] = 1
        return Y_onehot
