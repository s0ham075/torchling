import numpy as np
from hidden import Hidden


class NeuralNetwork():
    def __init__(self,layers=[],loss=None,):
        self.layers = layers
        if loss is not None:
            self.loss = loss.loss
            self.loss_prime = loss.loss_prime
        
    def add_layer(self,layer):
        self.layers.append(layer)

    def define_loss(self,loss):
        self.loss = loss.loss
        self.loss_prime = loss.loss_prime
    
    def predict(self,input):
        for layer in self.layers:
            input = layer.forward(input)
        
        return input
    
    def train(self,x_train,y_train,epochs=100,learning_rate=0.001):
        for _ in range(epochs):
            error = 0
            for i in range(len(x_train)):
                output = self.predict(x_train[i])
              
                error += self.loss(y_train[i],output)
                d_y = self.loss_prime(y_train[i],output)

                for layer in reversed(self.layers):
                    d_y = layer.backward(d_y,learning_rate)

            if _ % 10 == 0:
                print(f'Epoch: {_}, Error: {error/len(x_train)}')
                
