import numpy as np
from neural_network.layer import Layer

class Activation(Layer):
    def __init__(self,activation, activation_prime):
        self.activation = activation
        self.activation_prime = activation_prime
    
    def forward(self,input):
        self.input = input 
        return self.activation(self.input)
    
    def backward(self,grad,alpha = 0.01):
        return np.multiply(grad,self.activation_prime(self.input))