#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 28 21:03:45 2023

@author: soumensmacbookair
"""

import numpy as np

class Fully_Connected_Layer:

    def __init__(self, input_size, output_size):

        self.n0 = input_size
        self.n1 = output_size
        self.weight = np.random.randn(self.n0, self.n1)
        self.bias = np.random.randn(self.n1,1)
        self.a0 = None
        self.a1 = None

    def forward_prop(self, input_x):

        self.a0 = input_x
        self.a1 = np.dot(self.weight.T, self.a0) + self.bias
        return self.a1

    def backward_prop(self, dL_by_da1, learning_rate):
        
        m = self.a0.shape[1]
        dL_by_dW = np.dot(self.a0, dL_by_da1.T)
        dL_by_dB = dL_by_da1
        dL_by_da0 = np.dot(self.weight, dL_by_da1)
        
        assert dL_by_dW.shape == self.weight.shape
        assert dL_by_dB.shape == self.bias.shape

        self.weight = self.weight - learning_rate * dL_by_dW
        self.bias = self.bias - learning_rate * dL_by_dB

        return dL_by_da0

class Activation_Layer:

    def __init__(self, input_size, output_size, activation):

        self.n0 = input_size
        self.n1 = output_size
        self.a0 = None
        self.a1 = None
        self.activation = activation

    def forward_prop(self, input_x):

        self.a0 = input_x
        if self.activation == "sigmoid":
            self.a1 = np.exp(self.a0) / (1 + np.exp(self.a0))
        else:
            raise NotImplementedError()

        return self.a1

    def backward_prop(self, dL_by_da1, learning_rate):
        
        if self.activation == "sigmoid":
            dL_by_da0 = dL_by_da1 * self.a1 * (1-self.a1)
        else:
            raise NotImplementedError()

        return dL_by_da0

class Util():

    @staticmethod
    def mse(y_true, y_pred):
        return np.mean(np.power(y_true-y_pred, 2))

    @staticmethod
    def mse_prime(y_true, y_pred):
        return 2*(y_pred-y_true)/y_true.shape[0]

class FFNN():

    def __init__(self):

        self.layers = []
        self.loss_func = None

    def add_layer(self, layer_obj):
        
        self.layers.append(layer_obj)

    def predict(self, test_x):

        input_x = test_x
        for layer in self.layers:
            output_y = layer.forward_prop(input_x)
            input_x = output_y
        
        return output_y
    
    def update_parameter(self, dL_by_dY, learning_rate):

        dL_by_da1 = dL_by_dY
        for layer in list(reversed(self.layers)):
            dL_by_da0 = layer.backward_prop(dL_by_da1, learning_rate)
            dL_by_da1 = dL_by_da0
    
    def set_loss_function(self, loss_func):

        if loss_func == "mse":
            self.loss_func = Util.mse
            self.loss_func_prime = Util.mse_prime
        else:
            raise NotImplementedError()

    def fit(self, train_x, train_y, epoch=5, learning_rate=0.01):

        for ep in range(epoch):
            loss = 0
            for m in range(train_y.shape[1]):
                pred_y = self.predict(train_x[:,m].reshape((-1,1)))                       
                
                loss = loss + self.loss_func(train_y[:,m].reshape((-1,1)), pred_y)             
                dL_by_dY = self.loss_func_prime(train_y[:,m].reshape((-1,1)), pred_y)

                self.update_parameter(dL_by_dY, learning_rate)
        
            cost = loss / train_y.shape[1]
            print(f"Epoch {ep}/{epoch}: Cost = {cost}")
                
if __name__ == "__main__":

    network = FFNN()
    
    network.add_layer(Fully_Connected_Layer(2,3))
    network.add_layer(Activation_Layer(3,3,"sigmoid"))
    network.add_layer(Fully_Connected_Layer(3,3))
    network.add_layer(Activation_Layer(3,3,"sigmoid"))
    network.add_layer(Fully_Connected_Layer(3,1))

    train_x = np.array([[1,1,0,0],[1,0,1,0]])
    train_y = np.array([[0,1,1,0]])

    network.set_loss_function("mse")
    network.fit(train_x, train_y, epoch=1000, learning_rate=0.1)

    print(network.predict([[1],[1]]))
    print(network.predict([[1],[0]]))
    print(network.predict([[0],[1]]))
    print(network.predict([[0],[0]]))
