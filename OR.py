import numpy as np
from matplotlib import pyplot as plt
def sigmoid(z):
    return 1 / (1 + np.exp(-z))
def sse(val):
    return np.sum(np.square(np.subtract(np.mean(val),val)))
input = np.array([[0,0,1,1],[0,1,0,1],[-1,-1,-1,-1]]);
target = np.array([[0,1,1,1]])
W = np.random.rand(1,input.shape[0])
# print(input)
# print(target)
# print(W)
d_x = [-1, 0, 1, 2]
d_y = []
d_total = []
eta = 0.3
slope = -1 * W[0][0] / W[0][1]
for i in d_x:
    d_y.append(slope*i + (W[0][2]/W[0][1]))
d_total.append(d_y)
# print(d_y)
# print(d_total)
epoch = 1000
weight_list = []
error_list = []
print('Epoch\t\t\tError')
for i in range(epoch+1):
    y_k = np.dot(W,input)
    y = sigmoid(y_k)
    error = target - y
    error_list.append(sse(error[0]/4))
    delta = np.dot(error, np.transpose(input))
    W = W + eta*delta
    weight_list.append(W[0])
    if i!=0 and i%100 == 0:
        print(f"{i} \t\t {error_list[i]}")
        slope = -1*(weight_list[i][0]/weight_list[i][1])
        d_y = []
        for x in d_x:
            d_y.append(slope*x + (weight_list[i][2]/weight_list[i][1]))
        d_total.append(d_y)
print(np.round(y))
fig,ax = plt.subplots()
ax.plot([0,1,1],[1,0,1], 'bo')
ax.plot(0,0,'r+')
for i,x in enumerate(d_total):
    ax.plot(d_x, x, label=f"{i}")
ax.legend()
ax.set(xlabel="x-label", ylabel="y-label")
fig.suptitle("Decision Boundary")
plt.show()
fig, ax = plt.subplots()
ax.plot([i for i in range(epoch+1)], error_list)
plt.show()
