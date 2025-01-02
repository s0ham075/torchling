from layer import Layer
import numpy as np

class Hidden(Layer):
    def __init__(self,input_dim,output_dim,init_method= 'standard',bias_init=0):
        super().__init__(input_dim,output_dim)
        self.init_method = init_method # init_method is a string that specifies the initialization method to use
        if self.init_method == "standard":
            self.weights = np.random.randn(output_dim,input_dim) * 0.001
        
        if self.init_method == "xavier":
            self.weights = np.random.randn(output_dim,input_dim) * np.sqrt(1/input_dim)

        if self.init_method == "glorot":
            self.weights = np.random.randn(output_dim,input_dim) * np.sqrt(2/(input_dim+output_dim))
        
        if self.init_method == "he":
            self.weights = np.random.randn(output_dim,input_dim) * np.sqrt(2/input_dim)
        
        self.bias = np.ones((1,output_dim)) * bias_init
    
    def forward(self,input):
        self.input = input
        self.output = np.matmul(self.wieghts,input) + self.bias
        return self.output
    
    def backward(self,grad,alpha = 0.01):
        weights_grad = np.matmul(grad,self.input.T)
        self.wieghts -= alpha * weights_grad
        self.bias -= alpha * grad
        return np.matmul(self.weights.T,grad)