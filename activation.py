import numpy as np
from layer import Layer

class Activation(Layer):
    def __init__(self,activation, activation_prime):
        self.activation = activation
        self.activation_prime = activation_prime
    
    def forward(self,input):
        self.input = input 
        return self.activation(self.input)
    
    def backward(self,grad):
        return np.multiply(grad,self.activation_prime(self.input))