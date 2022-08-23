import numpy as np
from matplotlib import pyplot as plt

one = np.array([[0, 1, 1, 0, 0],
                [0, 0, 1, 0, 0],
                [0, 0, 1, 0, 0],
                [0, 0, 1, 0, 0],
                [0, 1, 1, 1, 0]])

two = np.array([[1, 1, 1, 1, 0],
                [0, 0, 0, 0, 1],
                [0, 1, 1, 1, 0],
                [1, 0, 0, 0, 0],
                [1, 1, 1, 1, 1]])

three = np.array([[1, 1, 1, 1, 0],
                  [0, 0, 0, 0, 1],
                  [0, 1, 1, 1, 0],
                  [0, 0, 0, 0, 1],
                  [1, 1, 1, 1, 0]])

four = np.array([[0, 0, 0, 1, 0],
                  [0, 0, 1, 1, 0],
                  [0, 1, 0, 1, 0],
                  [1, 1, 1, 1, 1],
                  [0, 0, 0, 1, 1]])
five = np.array([[1, 1, 1, 1, 1],
                  [1, 0, 0, 0, 0],
                  [1, 1, 1, 1, 0],
                  [0, 0, 0, 0, 1],
                  [1, 1, 1, 1, 0]])
D = np.array([[1, 0, 0, 0, 0],
              [0, 1, 0, 0, 0],
              [0, 0, 1, 0, 0],
              [0, 0, 0, 1, 0],
              [0, 0, 0, 0, 1]])

X = np.array([one, two, three, four, five])
# print(X)
def sigmoid(z):
    return 1 / (1+np.exp(-z))
def softmax(x):
    return np.exp(x) / np.sum(np.exp(x))
W1 = np.random.rand(50, 25)
W2 = np.random.rand(5,50)
alpha = 0.3
epoch = 1000

for i in range(epoch + 1):
    for j, val in enumerate(D):
        x = np.reshape(X[j], (25, 1))
        v1 = np.dot(W1, x)
        y1 = sigmoid(v1)
        v = np.dot(W2, y1)
        y = softmax(v)
        if i == (epoch - 1):
            print(np.round(y))
        d = np.reshape(val, (5, 1))
        e = d - y
        # delta = e * (1-y) * y
        delta = e
        dw2 = alpha * delta * np.transpose(y1)
        W2 = W2 + dw2

        e1 = np.dot(np.transpose(W2), delta)
        delta1 = e1 * y1 * (1 - y1)
        dw1 = alpha * delta1 * np.transpose(x)
        W1 = W1 + dw1
