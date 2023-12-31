#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 28 21:03:45 2023

@author: soumensmacbookair
"""

#%% Imports
import numpy as np

#%% Util class
class Util:

    @staticmethod
    def ReLU(x):
        return np.maximum(0, x)

    @staticmethod
    def ReLU_grad(x):
        return np.where(x >= 0, 1, 0)

    @staticmethod
    def sigmoid(x):
        return np.exp(x) / (1 + np.exp(x))

    @staticmethod
    def sigmoid_grad(x):
        return Util.sigmoid(x) * (1 - Util.sigmoid(x))

    @staticmethod
    def tanh(x):
        return (np.exp(2 * x) - 1) / (np.exp(2 * x) + 1)

    @staticmethod
    def tanh_grad(x):
        return 1 - (Util.tanh(x) ** 2)

    @staticmethod
    def mse(true_y, pred_y):
        # Mean is over the number of output neurons
        return np.mean((true_y - pred_y) ** 2, axis=0, keepdims=True)

    @staticmethod
    def mse_grad(true_y, pred_y):
        # We divide by number of output neurons
        return 2 * (true_y - pred_y) * (-1) / pred_y.shape[0]

#%% FF layer
class Fully_Connected_Layer:

    """
    a0 ------>  [FC Layer, [W] (n0 x n1) [b] (n1 x 1)] ------> a1
    (n0 x 1)                                                (n1 x 1)

    da0 <------  [FC Layer, [dW] (n0 x n1) [db] (n1 x 1)] <------ da1
    (n0 x 1)             [dx ~= dL/dx]                        (n1 x 1)
    """

    def __init__(self, input_size, output_size, init_factor_weight=0.01, init_factor_bias=0):

        self.type = "FC"
        self.n0 = input_size
        self.n1 = output_size
        self.W = init_factor_weight * np.random.randn(self.n0, self.n1)
        self.b = init_factor_bias * np.random.randn(self.n1, 1)
        self.a0 = None
        self.a1 = None
        self.da1 = None
        self.dW = None
        self.db = None
        self.da0 = None

    def forward_prop(self, input_x):

        self.a0 = input_x
        self.a1 = np.dot(self.W.T, self.a0) + self.b

        return self.a1

    def backward_prop(self, grad_output_y, learning_rate, clip_value):

        """For batch gradient descent, we want to take sum of all the gradients
        and then update once for the whole batch. For stochastic gradient descent, we
        will update the gradient after calculating the loss for a single training example
        and update the gradient immediately. For mini batch gradient descent or mini batch
        stochastic gradient descent, we will follow same but we will take mini batch instead
        of going for a full batch.
        """

        m = self.a0.shape[1] # Number of batch examples

        self.da1 = grad_output_y
        assert self.da1.shape == self.a1.shape, "Gradient of Output of layer must be of same shape with output"

        self.dW = np.dot(self.a0, self.da1.T) # Dot product automatically takes accummulated gradient for m > 1
        assert self.dW.shape == self.W.shape, "Gradient of Weight of layer must be of same shape with Weight"

        self.db = np.sum(self.da1, axis=1, keepdims=True) # np.sum has no effect if m = 1
        assert self.db.shape == self.b.shape, "Gradient of Bias of layer must be of same shape with Bias"

        self.da0 = np.dot(self.W,self.da1)
        assert self.da0.shape == self.a0.shape, "Gradient of Input of layer must be of same shape with Input"

        # Clip gradients for exploding gradient problem
        self.dW = np.clip(self.dW, -clip_value, clip_value)
        self.db = np.clip(self.db, -clip_value, clip_value)

        self.W = self.W - learning_rate * self.dW
        self.b = self.b - learning_rate * self.db

        return self.da0

#%% Activation layer
class Activation_Layer:

    """
    a0 ------>  [Activation Layer, No learnable parameters] ------> a1
    (n x 1)                                                    (n x 1)

    da0 <------  [Activation Layer, No learnable parameters] <------ da1
    (n x 1)             [dx ~= dL/dx]                           (n x 1)
    """

    def __init__(self, input_size, activation):

        self.type = "AC"
        self.n = input_size
        self.a0 = None
        self.a1 = None
        self.da1 = None
        self.da0 = None

        if activation == "relu":
            self.f = Util.ReLU
            self.f_grad = Util.ReLU_grad
        elif activation == "sigmoid":
            self.f = Util.sigmoid
            self.f_grad = Util.sigmoid_grad
        elif activation == "tanh":
            self.f = Util.tanh
            self.f_grad = Util.tanh_grad
        else:
            raise NotImplementedError()

    def forward_prop(self, input_x):

        self.a0 = input_x
        self.a1 = self.f(self.a0)

        return self.a1

    def backward_prop(self, grad_output_y):

        self.da1 = grad_output_y
        assert self.da1.shape == self.a1.shape, "Gradient of Output of layer must be of same shape with output"

        self.da0 = self.da1 * self.f_grad(self.a0)
        assert self.da0.shape == self.a0.shape, "Gradient of Input of layer must be of same shape with Input"

        return self.da0

#%% FFNN model
class FFNN:

    def __init__(self, loss_function):

        self.layers = []

        if loss_function == "mse":
            self.loss_func = Util.mse
            self.loss_func_grad = Util.mse_grad
        else:
            raise NotImplementedError()

    def add_layer(self, layer_obj):

        self.layers.append(layer_obj)

    def predict(self, input_x):

        for layer in self.layers:
            output_y = layer.forward_prop(input_x)
            input_x = output_y

        return output_y

    def update_parameter(self, input_x, true_y, learning_rate, clip_value):

        pred_y = self.predict(input_x) # Last layer output is the predicted value
        loss = self.loss_func(true_y, pred_y)
        grad_output_y = self.loss_func_grad(true_y, pred_y)

        for layer in reversed(self.layers):
            if layer.type == "FC":
                grad_output_y = layer.backward_prop(grad_output_y, learning_rate, clip_value)
            elif layer.type == "AC":
                grad_output_y = layer.backward_prop(grad_output_y)
            else:
                raise ValueError("Invalid layer type!")

        return loss

    def train(self, train_x, train_y, optimizer="FBGD", epoch=10, learning_rate=0.01, clip_value=5, batch_size=100):
        """ Optimizer Code
        FBGD - Full Batch Gradient Descent (Process full batch and update gradient 1 time per epoch)
        OGD - Online Gradient Descent (Process 1 example and update gradient m time per epoch, m = # of examples)
        SGD - Stochastic Gradient Descent (Process random 1 example and update gradient 1 time per epoch)
        MBSGD - Mini Batch Stochastic Gradient Descent (Process random 1 mini batch and update gradient 1 time per epoch)
        """

        for ep in range(epoch):

            if optimizer == "FBGD":
                loss = self.update_parameter(train_x, train_y, learning_rate, clip_value)
                print(f"Epoch: {ep}, Optimizer: {optimizer}, Loss: {loss.mean()}")

            elif optimizer == "MBSGD":
                m = train_y.shape[1]
                number_of_batch = np.ceil(m // batch_size)
                i = np.random.randint(0,number_of_batch)
                batch_train_x = train_x[:, batch_size * i: min(batch_size * (i+1) - 1, m - 1)]
                batch_train_y = train_y[:, batch_size * i: min(batch_size * (i+1) - 1, m - 1)]

                # if batch_train_x.shape[0] == batch_train_x.size:
                #     batch_train_x = batch_train_x.reshape(-1,1)
                #     batch_train_y = batch_train_y.reshape(-1,1)

                loss = self.update_parameter(batch_train_x, batch_train_y, learning_rate, clip_value)
                print(f"Epoch: {ep}, Optimizer: {optimizer}, Loss: {loss.mean()}")

            elif optimizer == "SGD":
                m = train_y.shape[1]
                i = np.random.randint(0, m)
                batch_train_x = train_x[:,i].reshape(-1,1)
                batch_train_y = train_y[:,i].reshape(-1,1)

                loss = self.update_parameter(batch_train_x, batch_train_y, learning_rate, clip_value)
                print(f"Epoch: {ep}, Optimizer: {optimizer}, Loss: {loss.mean()}")

            elif optimizer == "OGD":
                m = train_y.shape[1]
                loss = 0
                for i in range(m):
                    batch_train_x = train_x[:,i].reshape(-1,1)
                    batch_train_y = train_y[:,i].reshape(-1,1)

                    loss += self.update_parameter(batch_train_x, batch_train_y, learning_rate, clip_value)
                print(f"Epoch: {ep}, Optimizer: {optimizer}, Loss: {(loss/m).mean()}")

            else:
                raise ValueError("Invalid optimizer type!")

#%% Test code
if __name__ == "__main__":
    from sklearn.datasets import fetch_california_housing
    from sklearn.preprocessing import StandardScaler

    # Load the California housing dataset
    california_housing = fetch_california_housing()
    data = np.c_[california_housing.data, california_housing.target]

    # Separate features and target variable
    X = data[:, :-1]
    y = data[:, -1]

    # Use StandardScaler to normalize features
    scaler = StandardScaler()
    X_normalized = scaler.fit_transform(X)

    train_x = X_normalized.T
    train_y = y.reshape(-1,1).T

    network = FFNN("mse")

    network.add_layer(Fully_Connected_Layer(8,16))
    network.add_layer(Activation_Layer(16,"relu"))
    network.add_layer(Fully_Connected_Layer(16,8))
    network.add_layer(Activation_Layer(8,"relu"))
    network.add_layer(Fully_Connected_Layer(8,4))
    network.add_layer(Activation_Layer(4,"relu"))
    network.add_layer(Fully_Connected_Layer(4,2))
    network.add_layer(Activation_Layer(2,"relu"))
    network.add_layer(Fully_Connected_Layer(2,1))

    network.train(train_x, train_y, optimizer="FBGD",
        epoch=1000, learning_rate=0.001, clip_value=5, batch_size=100)



