from neural_network.layer import Layer
import numpy as np

class HiddenNoBias(Layer):
    def __init__(self,input_dim,output_dim,init_method= 'standard'):
        super().__init__(input_dim,output_dim)
        self.init_method = init_method # init_method is a string that specifies the initialization method to use
        if self.init_method == "standard":
            self.weights = np.random.randn(output_dim,input_dim) * 0.01
        
        if self.init_method == "xavier":
            self.weights = np.random.randn(output_dim,input_dim) * np.sqrt(1/input_dim)

        if self.init_method == "glorot":
            self.weights = np.random.randn(output_dim,input_dim) * np.sqrt(2/(input_dim+output_dim))
        
        if self.init_method == "he":
            self.weights = np.random.randn(output_dim,input_dim) * np.sqrt(2/input_dim)
        
    
    def forward(self,input):
        self.input = input
        if self.input.ndim == 0:  # Reshape if input is 1D
            self.input = self.input.reshape(-1, 1)
        
        print(self.weights.shape)
        print(self.input.shape)
        self.output = np.matmul(self.weights,self.input)
        return self.output
    
    def backward(self,grad,alpha = 0.01):
        weights_grad = np.matmul(grad,self.input.T)
        x = alpha * weights_grad
        self.weights -= alpha * weights_grad
        return np.matmul(self.weights.T,grad)