from Dense import *
from Sigmoid import *

class Neural_Network:
    def __init__(self,input_size,hidden_size1,hidden_size2,output_size):
        self.dense1 =Dense(input_size,hidden_size1)
        self.sigmoid1 = Sigmoid()
        self.dense2 = Dense(hidden_size1,hidden_size2)
        self.sigmoid2 = Sigmoid()
        self.dense3 = Dense(hidden_size2,output_size)
        self.sigmoid3 = Sigmoid()
    
    def Forward(self,input_data):
        z1 = self.dense1.Forward(input_data)
        a1 = self.sigmoid1.Forward(z1)
        z2 = self.dense2.Forward(a1)
        a2 = self.sigmoid2.Forward(z2)
        z3 = self.dense3.Forward(a2)
        a3 = self.sigmoid3.Forward(z3)

        return a3
    
    def Backward(self,grad_output,lr):
        op1 = self.sigmoid3.Backward(grad_output)
        op2 = self.dense3.Backward(op1,lr)
        op3 = self.sigmoid2.Backward(op2)
        op4 = self.dense2.Backward(op3,lr)
        op5 = self.sigmoid1.Backward(op4)
        op6 = self.dense1.Backward(op5,lr)

        return op6
