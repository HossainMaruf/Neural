import numpy as np
from matplotlib import pyplot as plt


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def sigmoid_prime(x):
    return x * (1.0 - x)


epochs = 5000
input_size, hidden_size, output_size = 3, 2, 1
learning_rate = 0.1

X = np.array([[0, 0, 1], [0, 1, 1], [1, 0, 1], [1, 1, 1]])
Y = np.array([[0], [1], [1], [0]])

w_hidden = np.random.uniform(size=(input_size, hidden_size))
w_output = np.random.uniform(size=(hidden_size, output_size))
hidden_weights = [0]
output_weights = [0]
error_list = []
momentum = 0.75
for epoch in range(epochs):
    hidden_output = sigmoid(np.dot(X, w_hidden))
    output = np.dot(hidden_output, w_output)
    error = Y - output

    # Backward Propagation
    delta = error * learning_rate
    dW = hidden_output.T.dot(delta)
    w_output += (dW + (momentum * output_weights[-1]))
    output_weights.append(dW)

    dH = delta.dot(w_output.T) * sigmoid_prime(hidden_output)
    w_hidden += X.T.dot(dH + (momentum * hidden_weights[-1]))
    hidden_weights.append(dH)

    error_list.append(np.sum(np.square(error)) / 4)

for i in X:
    hidden_output = sigmoid(np.dot(i, w_hidden))
    actual_output = np.dot(hidden_output, w_output)
    print(f"{i} : {actual_output}")

dc_bx = np.array([-1, 0, 1.5])
dc_bd = []
slop = -1 * w_hidden[0][0] / w_hidden[1][0]
dc_by = []
for i in dc_bx:
    dc_by.append(slop * i - w_hidden[2][0] / w_hidden[1][0])
dc_bd.append(dc_by)

slop = -1 * w_hidden[0][1] / w_hidden[1][1]
dc_by = []
for i in dc_bx:
    dc_by.append(slop * i - w_hidden[2][1] / w_hidden[1][1])
dc_bd.append(dc_by)

fig, ax = plt.subplots()
ax.plot([0, 1], [1, 0], 'r+')
ax.plot([0, 1], [0, 1], 'bo')
for i, x in enumerate(dc_bd):
    ax.plot(dc_bx, dc_bd[i], label=f"{i}th")
ax.set(xlabel='x-label', ylabel='y-label')
ax.legend()
fig.suptitle("Decision Boundary")
# ax.axis([-0.25, 1.5, -0.25, 1.5])

fig, ax = plt.subplots()
fig.suptitle("Error convergence curve")
ax.plot([i for i in range(epochs)], error_list)
plt.show()
