import numpy as np
import matplotlib.pyplot as plt  # Import Matplotlib for plotting
from neural_network import NeuralNetwork
from activations import ReLU
from hidden import Hidden
from loss import MSE
# Data preparation
x_train = np.arange(-3, 3, 0.1)
y_train = np.sin(x_train)

# Plotting
plt.figure(figsize=(8, 6))  # Set the figure size
# plt.plot(x_train, y_train, label='y = sin(x)', color='blue')

nn = NeuralNetwork(layers=[Hidden(1, 64, init_method='standard'), ReLU(), Hidden(64, 1, init_method='standard')], loss=MSE())

nn.train(x_train, y_train, epochs=1000, learning_rate=0.01)

y_pred = [nn.predict(x) for x in x_train]
y_pred = np.array([item.flatten() for item in y_pred])

x_train_new = np.arange(-5, 5, 0.1)
y_train_new = np.sin(x_train_new)
y_pred_new = [nn.predict(x) for x in x_train_new]
y_pred_new = np.array([item.flatten() for item in y_pred_new])

print(len(x_train_new))
print(len(y_pred_new))
plt.plot(x_train_new, y_pred_new, label='y = nn(x)', color='red')
plt.plot(x_train_new, y_train_new, label='y = sin(x)', color='blue')
# Show the plot
plt.show()
