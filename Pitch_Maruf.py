import numpy as np
from matplotlib import pyplot as plt

def sigmoid(z):
    return 1 / (1 + np.exp(-z))
def sse(val):
    return np.sum(np.square(np.subtract(np.mean(val),val)))
input = np.array([[0,0,1,1],[0,1,0,1],[-1,-1,-1,-1]]);

target_or = np.array([[0,1,1,1]])
target_nand = np.array([[1,1,1,0]])
target_and = np.array([[0,1,1,0]])

W1 = np.random.rand(1,input.shape[0])
W2 = np.random.rand(1, input.shape[0])
W3 = np.random.rand(1, input.shape[0])
# print(input)
# print(target)
# print(W)

d_x = [-1, 0, 1, 2]
d_y_or = []
d_y_nand = []
d_total_or = []
d_total_nand = []
eta = 0.3
slope1 = -1 * W1[0][0] / W1[0][1]
slope2 = -1 * W2[0][0] / W2[0][1]
for i in d_x:
    d_y_or.append(slope1*i + (W1[0][2]/W1[0][1]))
    d_y_nand.append(slope2*i + (W2[0][2]/W2[0][1]))
d_total_or.append(d_y_or)
d_total_nand.append(d_y_nand)
# print(d_y)
# print(d_total)

epoch = 1000
weight_list1 = []
weight_list2 = []
error_list1 = []
error_list2 = []
error_list3 = []
for i in range(epoch + 1):
    y_k1 = np.dot(W1, input)
    # print(W3.shape, y_k1.shape)
    y_k2 = np.dot(W2, input)
    # y_k3 = np.dot(np.transpose(W3), y_k1)
    y1 = sigmoid(y_k1)
    y2 = sigmoid(y_k2)
    mat = np.array([y1[0], y2[0], np.array([-1, -1, -1, -1])])
    y_k3 = np.dot(W3, mat)
    # print(y_k3.shape)
    y = sigmoid(y_k3)
    # print(y1.shape, y2.shape)
    # print(y)
    error1 = target_or - y1
    error2 = target_nand - y2
    error3 = target_and - y

    error_list1.append(sse(error1[0] / 4))
    error_list2.append(sse(error2[0] / 4))
    error_list3.append(sse(error3[0] / 4))
    delta1 = np.dot(error1, np.transpose(input))
    delta2 = np.dot(error2, np.transpose(input))
    delta3 = np.dot(error3, np.transpose(mat))
    W1 = W1 + eta * delta1
    W2 = W2 + eta * delta2
    W3 = W3 + eta * delta3
    weight_list1.append(W1[0])
    weight_list2.append(W2[0])
    if i != 0 and i % 100 == 0:
        # print(f"{i} \t\t {error_list[i]}")
        slope1 = -1 * (weight_list1[i][0] / weight_list1[i][1])
        slope2 = -1 * (weight_list2[i][0] / weight_list2[i][1])
        d_y_or = []
        d_y_nand = []
        for x in d_x:
            d_y_or.append(slope1 * x + (weight_list1[i][2] / weight_list1[i][1]))
            d_y_nand.append(slope2 * x + (weight_list2[i][2] / weight_list2[i][1]))
            # print(d_y_or)
        d_total_or.append(d_y_or)
        d_total_nand.append(d_y_nand)

print(np.round(y1))
print(np.round(y2))
print(y)
print(np.round(y))

fig,ax = plt.subplots()
ax.plot([0,1],[1,0], 'bo')
ax.plot([0,1],[0,1],'r+')
for i,x in enumerate(d_total_or):
    ax.plot(d_x, x, label=f"{i}")
for i,x in enumerate(d_total_nand):
    ax.plot(d_x, x, label=f"{i}")
# ax.legend()
ax.set(xlabel="x-label", ylabel="y-label")
fig.suptitle("Decision Boundary")
plt.show()
fig, ax = plt.subplots()
ax.plot([i for i in range(epoch+1)], error_list3)
plt.show()