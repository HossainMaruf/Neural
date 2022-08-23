import numpy as np
def sigmoid(z):
    return 1 / (1 + np.exp(-z))
X = np.array([[0,0,1],
     [0,1,1],
     [1,0,1],
     [1,1,1]])
D = np.array([[0, 1, 1, 0]])
W1 = np.random.rand(4,3)
W2 = np.random.rand(1,4)
# print(W1)
# print(W2)
# print(W1.shape)
# print(W2.shape)
ALPHA = 0.9
for i in range(2000):
    for j in range(D.shape[1]):
        x = np.reshape(X[j], (3, 1))
        v1 = np.dot(W1, x)
        y1 = sigmoid(v1)
        v2 = np.dot(W2, y1)
        y = sigmoid(v2)
        e = D[0, j] - y
        # learning rule (Delta)
        delta = e * y * (1 - y)
        dw2 = ALPHA * np.dot(delta, np.transpose(y1))
        W2 = W2 + dw2

        e1 = np.dot(np.transpose(W2), delta)
        delta = e1 * y1 * (1 - y1)
        dw1 = ALPHA * np.dot(delta, np.transpose(x))
        W1 = W1 + dw1

for j in range(D.shape[1]):
        x = np.reshape(X[j], (3,1))
        v1 = np.dot(W1,x)
        y1 = sigmoid(v1)
        v2 = np.dot(W2,y1)
        y = sigmoid(v2)
        print(np.round(y))