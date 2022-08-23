import numpy as np

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

X = np.array([[0,0,1],[0,1,1],[1,0,1],[1,1,1]])
D = np.array([[0,0,1,1]])
W = np.random.rand(1,3)
ALPHA = 0.9
# print(X,D,W)

for i in range(7):
    for j in range(D.shape[1]):
        x = np.reshape(X[j], (3,1))
        v = np.dot(W,x)
        y = sigmoid(v)
        d = D[:, j]
        e = d - y
        delta = y * (1-y) * e
        dw = ALPHA * delta * x
        W = W + np.transpose(dw)

for i in range(D.shape[1]):
    x = np.reshape(X[i], (3,1))
    v = np.dot(W,x)
    y = sigmoid(v)
    print(np.round(y))