import numpy as np
import matplotlib.pyplot as plt
import sys
from pathlib import Path

# Add the project root directory to PYTHONPATH
sys.path.append(str(Path(__file__).resolve().parents[1]))
  # Import Matplotlib for plotting
from neural_network import NeuralNetwork
from neural_network.activations import ReLU
from neural_network.hidden import Hidden   
from neural_network.hidden_no_bias import HiddenNoBias
from neural_network.loss import MSE
# Data preparation
x_train = np.arange(-3, 3, 0.001)
y_train = np.sin(x_train)

# Plotting
plt.figure(figsize=(8, 6))  # Set the figure size


nn = NeuralNetwork(layers=[Hidden(1, 8, init_method='glorot'), ReLU(), HiddenNoBias(8, 1, init_method='glorot')], loss=MSE())

# nn.train(x_train, y_train, epochs=200)
weights_layer1 = np.array([
    [0.38451323],
    [-0.24923242],
    [0.68425363],
    [0.38631511],
    [-0.42454637],
    [-0.49184521],
    [-0.19948901],
    [-0.5889988]
])

biases_layer1 = np.array([
    [0.81057517],
    [-0.34714969],
    [-0.94844159],
    [-0.92133055],
    [-0.59156027],
    [-0.69207904],
    [0.34630943],
    [1.00818593]
])

weights_layer2 = np.array([
    [0.83510696],
    [0.44998368],
    [-1.10021663],
    [-1.19525465],
    [0.7734859],
    [0.80762141],
    [-0.30300423],
    [-0.5536479]
])

nn.layers[0].weights = weights_layer1
nn.layers[0].bias = biases_layer1
nn.layers[2].weights = weights_layer2.T


y_pred = [nn.predict(x) for x in x_train]
y_pred = np.array([item.flatten() for item in y_pred])


plt.plot(x_train, y_train, label='y = sin(x)', color='blue')
plt.plot(x_train, y_pred, label='y = nn(x)', color='red')

print("1 - ",nn.layers[0].weights)
print("bias - ",nn.layers[0].bias)
print(nn.layers[2].weights)
# x_train_new = np.arange(-5, 5, 0.1)
# y_train_new = np.sin(x_train_new)
# y_pred_new = [nn.predict(x) for x in x_train_new]
# y_pred_new = np.array([item.flatten() for item in y_pred_new])

# plt.plot(x_train_new, y_pred_new, label='y = nn(x)', color='red')
# plt.plot(x_train_new, y_train_new, label='y = sin(x)', color='blue')
# # Show the plot
plt.show()
