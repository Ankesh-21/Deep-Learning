import numpy as np

class Dense:
    def __init__(self,input_size,output_size):
        self.weights = np.random.randn(output_size,input_size)
        self.bias = np.zeros((output_size,1))
        self.bita = 0.9

        self.v_weights = np.zeros_like(self.weights)
        self.v_bias = np.zeros_like(self.bias)

    def Forward(self,input_data):
        self.inputs = input_data
        return np.dot(self.weights,self.inputs) + self.bias

    def Backward(self,grad_output,lr):
        grad_weights = np.dot(grad_output , self.inputs.T)
        grad_inputs = np.dot(self.weights.T,grad_output)
        grad_bias = np.sum(grad_output,axis = 1,keepdims = True)

        self.v_weights = (self.bita * self.v_weights) + (1- self.bita)*grad_weights
        self.v_bias = (self.bita * self.v_bias) + (1- self.bita)*grad_bias

        self.weights -= lr * self.v_weights
        self.bias -= lr * self.v_bias
        return grad_inputs

