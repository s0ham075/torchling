import numpy as np
from hidden import Hidden


class NeuralNetwork():
    def __init__(self,layers=[],loss=None,loss_prime=None):
        self.layers = layers
        self.loss = loss
        self.loss_prime = loss_prime
        
    def add_layer(self,layer):
        self.layers.append(layer)
    
    def predict(self,input):
        for layer in self.layers:
            input = layer.forward(input)
        
        return input
    
    def train(self,x_train,y_train,epochs=100,learning_rate=0.1):
        for _ in range(epochs):
            error = 0
            for i in range(len(x_train)):
                output = self.predict(x_train[i])
              
                error += self.loss(y_train[i],output)
                
                for layer in reversed(self.layers):
                    error = layer.backward(self.loss_prime(y_train[i],output),learning_rate)
                
            if _ % 10 == 0:
                print(f'Epoch: {_}, Error: {error/len(x_train)}')
