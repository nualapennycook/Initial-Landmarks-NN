import matplotlib.pyplot as plt
import numpy as np

def elu(x):
    y = np.empty(len(x))
    for i in range(len(x)):
        if x[i] >= 0:
            y[i] = x[i]
        else:
            y[i] =  np.exp(x[i]) -1
    return y

def relu(x):
    y = np.empty(len(x))
    for i in range(len(x)):
        y[i] = max(0, x[i])
    return y


fig, axs = plt.subplots(2, 2)

x = np.linspace(-10, 10, 100)

axs[0, 0].plot(x, 1/(1+np.exp(-x)))
axs[0, 0].set_title('Sigmoid Function')
axs[0, 1].plot(x, np.tanh(x))
axs[0, 1].set_title('Tanh')
axs[1, 0].plot(x, relu(x))
axs[1, 0].set_title('ReLU - Rectified Linear Units')
axs[1, 1].plot(x, elu(x))
axs[1, 1].set_title('ELU - Exponential Linear Units')

# Hide x labels and tick labels for top plots and y ticks for right plots.
for ax in axs.flat:
    ax.label_outer()

plt.show()

