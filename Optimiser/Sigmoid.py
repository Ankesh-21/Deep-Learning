import numpy as np

class Sigmoid:

    def Forward(self,x):
        self.outputs = 1 / (1 + np.exp(-x))
        return self.outputs

    def Backward(self,grad_outputs):
        grad_sig = self.outputs * (1 - self.outputs)
        return grad_sig * grad_outputs