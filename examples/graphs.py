import numpy as np
import matplotlib.pyplot as plt

# Provided weights and biases
weights_layer1 = np.array([
    [0.38451323], [-0.24923242], [0.68425363], [0.38631511],
    [-0.42454637], [-0.49184521], [-0.19948901], [-0.5889988]
])
biases_layer1 = np.array([
    [0.81057517], [-0.34714969], [-0.94844159], [-0.92133055],
    [-0.59156027], [-0.69207904], [0.34630943], [1.00818593]
])
weights_layer2 = np.array([
    [0.83510696], [0.44998368], [-1.10021663], [-1.19525465],
    [0.7734859], [0.80762141], [-0.30300423], [-0.5536479]
])

# Generate input data
x = np.linspace(-10, 10, 200)

def relu(x):
    return np.maximum(0, x)

# Define plotting functions
def plot1(weights, biases, x): return relu(weights[0]*x + biases[0])
def plot2(weights, biases, x): return relu(weights[1]*x + biases[1])
def plot3(weights, biases, x): return relu(weights[2]*x + biases[2])
def plot4(weights, biases, x): return relu(weights[3]*x + biases[3])
def plot5(weights, biases, x): return relu(weights[4]*x + biases[4])
def plot6(weights, biases, x): return relu(weights[5]*x + biases[5])
def plot7(weights, biases, x): return relu(weights[6]*x + biases[6])
def plot8(weights, biases, x): return relu(weights[7]*x + biases[7])

def combined(x):
    y1 = weights_layer2[0] * (plot1(weights_layer1, biases_layer1, x))
    y2 = weights_layer2[1] * (plot2(weights_layer1, biases_layer1, x))
    y3 = weights_layer2[2] * (plot3(weights_layer1, biases_layer1, x))
    y4 = weights_layer2[3] * (plot4(weights_layer1, biases_layer1, x))
    y5 = weights_layer2[4] * (plot5(weights_layer1, biases_layer1, x))
    y6 = weights_layer2[5] * (plot6(weights_layer1, biases_layer1, x))
    y7 = weights_layer2[6] * (plot7(weights_layer1, biases_layer1, x))
    y8 = weights_layer2[7] * (plot8(weights_layer1, biases_layer1, x))
    return y1 + y2 + y3 + y4 + y5 + y6 + y7 + y8

# Create subplots
fig, axs = plt.subplots(3, 3, figsize=(15, 15))
axs = axs.ravel()

# Plot individual functions
plot_funcs = [plot1, plot2, plot3, plot4, plot5, plot6, plot7, plot8]
for i, func in enumerate(plot_funcs):
    axs[i].plot(x, func(weights_layer1, biases_layer1, x))
    axs[i].set_title(f'Function {i+1}')
    axs[i].grid(True)
    axs[i].set_xlim([-10, 10])
    axs[i].set_ylim([-10, 10])
    axs[i].axhline(y=0, color='k', linestyle='-', alpha=0.3)
    axs[i].axvline(x=0, color='k', linestyle='-', alpha=0.3)

# Plot combined function and sine wave
axs[8].plot(x, combined(x), 'r-', label='Combined')
axs[8].plot(x, np.sin(x), 'b--', label='Sine Wave')
axs[8].set_title('Combined Function and Sine Wave')
axs[8].grid(True)
axs[8].set_xlim([-10, 10])
axs[8].set_ylim([-10, 10])
axs[8].axhline(y=0, color='k', linestyle='-', alpha=0.3)
axs[8].axvline(x=0, color='k', linestyle='-', alpha=0.3)
axs[8].legend()

plt.tight_layout()
plt.show()