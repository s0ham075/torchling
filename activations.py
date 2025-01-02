import numpy as np
from activation import Activation

class Sigmoid(Activation):
    def __init__(self):
        def sigmoid(x):
            return 1/(1+ np.exp(-x))
        
        def sigmoid_prime(x):
            return sigmoid(x) * (1 - sigmoid(x))
        
        super().__init__(sigmoid,sigmoid_prime)


class Tanh(Activation):
    def __init__(self):
        def tanh(x):
            return np.tanh(x)
        
        def tanh_prime(x):
            return 1 - np.tanh(x)**2
        
        super().__init__(tanh,tanh_prime)

class ReLU(Activation):
    def __init__(self):
        def relu(x):
            return np.maximum(0,x)
        
        def relu_prime(x):
            return 1 if x > 0 else 0
        
        super().__init__(relu,relu_prime)

class LeakyReLU(Activation):
    def __init__(self, a=0.01):
        def leaky_relu(x):
            return np.maximum(a*x,x)

        def leaky_relu_prime(x):
            return 1 if x > 0 else a  

        super().__init__(leaky_relu,leaky_relu_prime)      