import numpy as np
import matplotlib.pyplot as plt
# input = np.array([[0, 0, 1, 1], [0, 1, 0, 1], [-1, -1, -1, -1]])
# target = np.array([[0, 0, 0, 1]])
input = np.array([[-1, -1, 1, 1], [-1, 1, -1, 1], [-1, -1, -1, -1]])
target = np.array([[-1, -1, -1, 1]])
weight_list = []
error_list = []
# print(input)
# print(input.shape)
# print(target)
# print(target.shape)
W = np.random.rand(1, input.shape[0])
# print(W)
# W = np.array([[0.79752527 0.17863946 0.62393269]])
# W = np.array([[0.13852945 0.47320946 0.78652735]])
# print(np.dot(W,input))
#print(W[0][0])
#print(W[0][1])
# Initial Line
d_x = [-1,0,1,2,3,4]
d_y = []
d_total = []
slope = -1 * W[0][0] / W[0][1]
for i in d_x:
    d_y.append(slope*i + (W[0][2] / W[0][1]))
d_total.append(d_y)
# print(d_y)
# print(d_total)
def sigmoid(z):
    return (1-np.exp(-z)) / (1+np.exp(-z))
def sse(val):
    return np.sum(np.square(np.subtract(np.mean(val),val)))
eta = 0.7
epoch = 1000
for i in range(epoch + 1):
    y_k = np.dot(W, input)
    y = sigmoid(y_k)
    error = target - y
    # f_d = 0.5 * (error * (1 - np.square(y)))
    delta = np.dot(error, np.transpose(input))
    W = W + eta * delta
    weight_list.append(W[0])
    error_list.append(sse(error[0] / 4))
    if i != 0 and i % 100 == 0:
        # print(f"{i} \t\t {error_list[i]}")
        slop = -1 * (weight_list[i][0] / weight_list[i][1])
        d_y = []
        for x in d_x:
            d_y.append(slop * x + weight_list[i][2] / weight_list[i][1])
        d_total.append(d_y)
# print(error)
# print(y)
print(np.round(y))
# print(len(d_total))
# print((d_x,d_total[i]) for i in range(5))
fig, ax = plt.subplots()
ax.plot([-1,-1,1],[-1,1,-1], 'bo')
ax.plot(1,1, 'r+')
for i,x in enumerate(d_total):
    ax.plot(d_x,x,label=f"{i}")
    ax.legend()
ax.set(xlabel='x', ylabel='y')
ax.axis([-5, 5, -3, 3])
plt.show()
fig, ax = plt.subplots()
fig.suptitle('Error convergence curve')
ax.set(xlabel='No. of iteration', ylabel='Error')
ax.plot([i for i in range(epoch+1)], error_list)
plt.show()
