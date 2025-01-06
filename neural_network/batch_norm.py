from neural_network.layer import Layer
import numpy as np

class Batch_Norm(Layer):
    def __init__(self,output_dim,epsilon=1e-5):
        super().__init__(output_dim,output_dim)
        self.epsilon = epsilon
        self.gamma = np.ones((output_dim,1))
        self.beta = np.zeros((output_dim,1))
    
    def forward(self,input):
        self.input = input
        self.mean = np.mean(input, axis=0, keepdims=True)
        self.variance = np.var(input, axis=0, keepdims=True)
        self.x_norm = (input - self.mean) / np.sqrt(self.variance + self.epsilon)
        self.output = self.gamma * self.x_norm + self.beta
        return self.output
    
    def backward(self,grad,alpha = 0.01):
        gamma_grad = np.sum(grad * self.x_norm, axis=0, keepdims=True)
        beta_grad = np.sum(grad, axis=1, keepdims=True)
    
        dx_norm = grad * self.gamma
        dvariance = np.sum(dx_norm * (self.input - self.mean) * -0.5 * np.power(self.variance + self.epsilon, -1.5), axis=0, keepdims=True)
        dmean = np.sum(-dx_norm / np.sqrt(self.variance + self.epsilon), axis=0, keepdims=True) + dvariance * np.sum(-2 * (self.input - self.mean), axis=0, keepdims=True) / grad.shape[0]
        dx = dx_norm / np.sqrt(self.variance + self.epsilon) + 2 * dvariance * (self.input - self.mean) / grad.shape[0] + dmean / grad.shape[0]

    # Update gamma and beta parameters
        self.gamma -= 0.01 * gamma_grad
        self.beta -= 0.01 * beta_grad
        return dx