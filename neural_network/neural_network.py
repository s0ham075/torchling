import numpy as np
from neural_network.hidden import Hidden


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
    

    def train(self, x_train, y_train, epochs=100, learning_rate=0.01, batch_size=32):
        self.error = 1
        for _ in range(epochs):
            if self.error <= 0.001:
                learning_rate = 0.001

            if self.error <= 0.0001:
                learning_rate = 0.0001

            if self.error <= 0.00001:
                learning_rate = 0.00001

            # Shuffle the data at the start of each epoch
            indices = np.arange(len(x_train))
            np.random.shuffle(indices)
            x_train = x_train[indices]
            y_train = y_train[indices]

            # Split data into batches
            x_batches = np.array_split(x_train, len(x_train) // batch_size)
            y_batches = np.array_split(y_train, len(y_train) // batch_size)

            error = 0
            for x_batch, y_batch in zip(x_batches, y_batches):
                output = self.predict(x_batch)

                batch_error = self.loss(y_batch, x_batch)
                d_y = self.loss_prime(y_batch, output)

                for layer in reversed(self.layers):
                    d_y = layer.backward(d_y, learning_rate)

                error += batch_error

            self.error = error

            if _ % 10 == 0:
                print(f'Epoch: {_}, Error: {error / len(x_train)}')
