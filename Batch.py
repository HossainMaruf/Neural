import numpy as np

def sigmoid(z):
    return 1 / (1 + np.exp(-z))
X = np.array([[0, 0, 1], [0, 1, 1], [1, 0, 1], [1, 1, 1]])
D = np.array([[0, 0, 1, 1]])

W = np.random.rand(1,3)
W = np.array([[0.91828964, 0.59173   , 0.33396307]])

# print(X.shape)
# print(D.shape)
# print(W.shape)
ALPHA = 0.9
dWsum = np.zeros((3,1))

for i in range(5):
    for j in range(D.shape[1]):
        x = np.reshape(X[j], (3, 1))
        d = D[:, j]
        v = np.dot(W, x)
        y = sigmoid(v)
        e = d - y
        delta = y * (1 - y) * e
        dW = ALPHA * delta * x
        dWsum = dWsum + dW

    dWavg = np.divide(dWsum, 4)
    W = W + np.transpose(dWavg)

for i in range(D.shape[1]):
    x = np.reshape(X[i], (3,1))
    v = np.dot(W,x)
    y = sigmoid(v)
    print(y)
    print(np.round(y))