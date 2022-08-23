import numpy as np
from matplotlib import pyplot as plt

alpha = 0.7
epoch = 1000
N = 4

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def DeltaSGD(W,X,D):
    for i in range(D.shape[1]):
        x = np.reshape(X[i], (3,1))
        v = np.dot(W,x)
        y = sigmoid(v)
        # print(y.shape)
        e = D[:,i] - y
        delta = e * y * (1-y)
        dw = alpha * delta * x
        W = W + np.transpose(dw)
        # print(dw.shape)
    return W

def DeltaBatch(W,X,D):
    dwSum = np.zeros((3,1))
    for i in range(D.shape[1]):
        x = np.reshape(X[i],(3,1))
        v = np.dot(W,x)
        y = sigmoid(v)
        e = D[:,i] - y
        delta = e * y * (1-y)
        dw = alpha * delta * x
        dwSum = dwSum + dw
    dwAvg = np.divide(dwSum,4)
    W = W + np.transpose(dwAvg)
    return W

W1 = np.random.rand(1,3)
W2 = W1
# print(W)
X = np.array([[0, 0, 1], [0, 1, 1], [1, 0, 1], [1, 1, 1]])
D = np.array([[0, 0, 1, 1]])
# print(X,D)
error_list1 = np.zeros((epoch,1))
error_list2 = np.zeros((epoch,1))

for i in range(epoch):
    W1 = DeltaSGD(W1, X, D)
    W2 = DeltaBatch(W2, X, D)
    es1 = 0
    es2 = 0
    for k in range(N):
        x = np.reshape(X[k], (3, 1))

        v1 = np.dot(W1, x)
        y1 = sigmoid(v1)
        e1 = D[:, k] - y1
        es1 = es1 + np.square(e1)
        # print(es1)
        v2 = np.dot(W2, x)
        y2 = sigmoid(v2)
        e2 = D[:, k] - y2
        es2 = es2 + np.square(e2)
    error_list1[i] = es1 / N
    error_list2[i] = es2 / N

plt.plot(error_list1, 'r')
plt.plot(error_list2, 'b:')
plt.xlabel('Epoch')
plt.ylabel('Training Error')
plt.legend(['SGD', 'Batch'])
plt.show()