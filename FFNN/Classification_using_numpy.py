#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 01 21:03:45 2023

@author: soumensmacbookair
"""

#%% Imports
import numpy as np
import pickle

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
    def softmax(x):
        # assert x.shape[1] == 1, "x shape must be (n,1)"
        exp_x = np.exp(x)
        return exp_x / np.sum(exp_x, axis=0, keepdims=True)

    @staticmethod
    def softmax_grad(x):
        assert x.shape[1] == 1, "x shape must be (n,1)"
        n = x.shape[0]

        sm_x = Util.softmax(x)
        sm_grad_x = -1 * np.dot(sm_x, sm_x.T)
        np.fill_diagonal(sm_grad_x, sm_x.flatten() * (1 - sm_x.flatten()))

        return sm_grad_x

    @staticmethod
    def cross_entropy(true_p, pred_p):
        assert true_p.shape[1] == 1, "true_p is onehot encoded of (k,1) vector where k is number of class"
        assert true_p.shape == pred_p.shape, "pred_p is a softmax probabilities with same same as of true_p"

        return -1 * np.sum(true_p * np.log(pred_p))

    @staticmethod
    def cross_entropy_grad(true_p, pred_p):
        assert true_p.shape[1] == 1, "true_p is onehot encoded of (k,1) vector where k is number of class"
        assert true_p.shape == pred_p.shape, "pred_p is a softmax probabilities with same same as of true_p"

        return -1 * true_p / pred_p

    @staticmethod
    def calc_accuracy(true_probability, pred_probability):

        true_label = np.argmax(true_probability, axis=0, keepdims=True)
        pred_label = np.argmax(pred_probability, axis=0, keepdims=True)
        accuracy = (np.where(true_label == pred_label, 1, 0)).mean() * 100

        return accuracy

#%% FF layer
class Fully_Connected_Layer:

    """
    a0 ------>  [FC Layer, [W] (n0 x n1) [b] (n1 x 1)] ------> a1
    (n0 x 1)                                                (n1 x 1)

    da0 <------  [FC Layer, [dW] (n0 x n1) [db] (n1 x 1)] <------ da1
    (n0 x 1)             [dx ~= dL/dx]                        (n1 x 1)
    """

    def __init__(self, input_size, output_size):

        self.type = "FC"
        self.n0 = input_size
        self.n1 = output_size
        self.W = np.random.randn(self.n0, self.n1) * np.sqrt(2 / (self.n0 + self.n1))
        self.b = 0.01 * np.random.randn(self.n1, 1)
        self.a0 = np.zeros((self.n0,1))
        self.a1 = np.zeros((self.n1,1))
        self.da1 = np.zeros_like(self.a1)
        self.dW = np.zeros_like(self.W)
        self.db = np.zeros_like(self.b)
        self.da0 = np.zeros_like(self.a0)
        self.activation = None

    def forward_prop(self, input_x):

        self.a0 = input_x
        self.a1 = np.dot(self.W.T, self.a0) + self.b

        return self.a1

    def backward_prop(self, grad_output_y, learning_rate, clip_value, batch_size, update):

        """For batch gradient descent, we want to take sum of all the gradients
        and then update once for the whole batch. For stochastic gradient descent, we
        will update the gradient after calculating the loss for a single training example
        and update the gradient immediately. For mini batch gradient descent or mini batch
        stochastic gradient descent, we will follow same but we will take mini batch instead
        of going for a full batch.
        """

        self.da1 = grad_output_y
        assert self.da1.shape == self.a1.shape, "Gradient of Output of layer must be of same shape with output"

        self.dW += np.dot(self.a0, self.da1.T) # Dot product automatically takes accummulated gradient for m > 1
        assert self.dW.shape == self.W.shape, "Gradient of Weight of layer must be of same shape with Weight"

        self.db += np.sum(self.da1, axis=1, keepdims=True) # np.sum has no effect if m = 1
        assert self.db.shape == self.b.shape, "Gradient of Bias of layer must be of same shape with Bias"

        self.da0 = np.dot(self.W,self.da1)
        assert self.da0.shape == self.a0.shape, "Gradient of Input of layer must be of same shape with Input"

        if update:
            # Clip gradients for exploding gradient problem
            self.dW = np.clip(self.dW, -clip_value * batch_size, clip_value * batch_size)
            self.db = np.clip(self.db, -clip_value * batch_size, clip_value * batch_size)

            # Update the gradients for this example
            self.W = self.W - learning_rate * self.dW / batch_size
            self.b = self.b - learning_rate * self.db / batch_size

            # Clear the gradients for this example once update is over
            self.dW = np.zeros_like(self.W)
            self.db = np.zeros_like(self.b)

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
        self.n0 = input_size
        self.n1 = input_size
        self.W = None
        self.b = None
        self.a0 = np.zeros((self.n0, 1))
        self.a1 = np.zeros_like(self.a0)
        self.da1 = np.zeros_like(self.a1)
        self.da0 = np.zeros_like(self.a0)
        self.activation = activation

    def forward_prop(self, input_x):

        if self.activation == "relu":
            f = Util.ReLU
        elif self.activation == "sigmoid":
            f = Util.sigmoid
        elif self.activation == "tanh":
            f = Util.tanh
        elif self.activation == "softmax":
            f = Util.softmax
        else:
            raise NotImplementedError()

        self.a0 = input_x
        self.a1 = f(self.a0)

        return self.a1

    def backward_prop(self, grad_output_y):

        if self.activation == "relu":
            f_grad = Util.ReLU_grad
        elif self.activation == "sigmoid":
            f_grad = Util.sigmoid_grad
        elif self.activation == "tanh":
            f_grad = Util.tanh_grad
        elif self.activation == "softmax":
            f_grad = Util.softmax_grad
        else:
            raise NotImplementedError()

        self.da1 = grad_output_y
        assert self.da1.shape == self.a1.shape, "Gradient of Output of layer must be of same shape with output"

        if f_grad == Util.softmax_grad:
            # For softmax, grad is matrix so dot product is needed for a single example
            self.da0 = np.dot(f_grad(self.a0), self.da1)
        else:
            # For other than softmax, we need elementwise mult for many examples and
            # for softmax, we don't allow multiple examples as grad will be 3rd order tensor
            self.da0 = f_grad(self.a0) * self.da1

        assert self.da0.shape == self.a0.shape, "Gradient of Input of layer must be of same shape with Input"

        return self.da0

#%% FFNN model
class FFNN:

    def __init__(self, loss_function):

        self.layers = []

        if loss_function == "cross_entropy":
            self.loss_func = Util.cross_entropy
            self.loss_func_grad = Util.cross_entropy_grad
        else:
            raise NotImplementedError()

    def add_layer(self, layer_obj):

        self.layers.append(layer_obj)

    def save_model(self, filename):

        model_params = {"layers": []}
        layer_id = 1
        for layer in self.layers:
            if layer.type == "FC":
                layer_name = f"Layer Number: {layer_id} [Fully Connected]"
                layer_id += 1
            else:
                layer_name = f"Layer Number: {layer_id} [Activation]"

            layer_params = {
                "layer_type": layer.type,
                "layer_name": layer_name,
                "Weight": layer.W,
                "Bias": layer.b,
                "Input_size": layer.n0,
                "Output_size": layer.n1,
                "Activation": layer.activation
            }
            model_params['layers'].append(layer_params)

        with open(filename, 'wb') as file:
            pickle.dump(model_params, file)

    def load_model(self, filename):

        with open(filename, 'rb') as file:
            model_params = pickle.load(file)

        for i,layer in enumerate(model_params["layers"]):
            if layer["layer_type"] == "FC":
                self.add_layer(Fully_Connected_Layer(layer["Input_size"],layer["Output_size"]))
            elif layer["layer_type"] == "AC":
                self.add_layer(Activation_Layer(layer["Input_size"],layer["Activation"]))
            else:
                raise ValueError("Invalid layer type!")

            self.layers[i].W = layer["Weight"]
            self.layers[i].b = layer["Bias"]

    def predict(self, input_x, label=False):

        for layer in self.layers:
            output_y = layer.forward_prop(input_x)
            input_x = output_y

        if label:
            return np.argmax(output_y, axis=0, keepdims=True)
        else:
            return output_y

    def update_parameter(self, input_x, true_y, learning_rate, clip_value, batch_size, update):

        pred_y = self.predict(input_x) # Last layer output is the predicted value
        loss = self.loss_func(true_y, pred_y)
        grad_output_y = self.loss_func_grad(true_y, pred_y)

        for layer in reversed(self.layers):
            if layer.type == "FC":
                grad_output_y = layer.backward_prop(grad_output_y, learning_rate, clip_value, batch_size, update)
            elif layer.type == "AC":
                grad_output_y = layer.backward_prop(grad_output_y)
            else:
                raise ValueError("Invalid layer type!")

        return loss

    def train(self, train_x, train_y, optimizer="SGD", epoch=10000, learning_rate=0.01, clip_value=5, minibatch_size=100):
        """ Optimizer Code
        No accummulated based gradient descent is implemented as it needs 3rd order tensor
        SGD - Stochastic Gradient Descent (Process random 1 example and update gradient 1 time per epoch)
        MBSOGD - Mini Batch Stochastic (Online) Gradient Descent (Process random 1 mini batch and update gradient 1 time per b batch examples per epoch)
        MBSGD - Mini Batch Stochastic Gradient Descent (Process random 1 mini batch and update cumulative gradient 1 time per epoch for all b examples)

        """

        for ep in range(epoch):

            if optimizer == "MBSOGD":
                m = train_y.shape[1]
                number_of_batch = np.ceil(m // minibatch_size)
                i = np.random.randint(0,number_of_batch)
                minibatch_train_x = train_x[:, minibatch_size * i: min(minibatch_size * (i+1) - 1, m - 1)]
                minibatch_train_y = train_y[:, minibatch_size * i: min(minibatch_size * (i+1) - 1, m - 1)]

                loss = 0
                for j in range(minibatch_train_y.shape[1]):
                    batch_train_x = minibatch_train_x[:,j].reshape(-1,1)
                    batch_train_y = minibatch_train_y[:,j].reshape(-1,1)

                    loss += self.update_parameter(batch_train_x, batch_train_y, learning_rate, clip_value, batch_size=1, update=True)

                if ep % 10 == 0:
                    pred_train_y = self.predict(train_x)
                    train_accuracy = Util.calc_accuracy(train_y, pred_train_y)
                    print(f"Epoch: {ep}, Optimizer: {optimizer}, Loss: {loss/minibatch_train_y.shape[1]}, Train Accuracy: {train_accuracy}")

            elif optimizer == "SGD":
                m = train_y.shape[1]
                i = np.random.randint(0, m)
                batch_train_x = train_x[:,i].reshape(-1,1)
                batch_train_y = train_y[:,i].reshape(-1,1)

                loss = self.update_parameter(batch_train_x, batch_train_y, learning_rate, clip_value, batch_size=1, update=True)

                if ep % 100 == 0:
                    pred_train_y = self.predict(train_x)
                    train_accuracy = Util.calc_accuracy(train_y, pred_train_y)
                    print(f"Epoch: {ep}, Optimizer: {optimizer}, Loss: {loss}, Train Accuracy: {train_accuracy}")

            elif optimizer == "MBSGD":
                m = train_y.shape[1]
                number_of_batch = np.ceil(m // minibatch_size)
                i = np.random.randint(0,number_of_batch)
                minibatch_train_x = train_x[:, minibatch_size * i: min(minibatch_size * (i+1) - 1, m - 1)]
                minibatch_train_y = train_y[:, minibatch_size * i: min(minibatch_size * (i+1) - 1, m - 1)]

                loss = 0
                for j in range(minibatch_train_y.shape[1]):
                    batch_train_x = minibatch_train_x[:,j].reshape(-1,1)
                    batch_train_y = minibatch_train_y[:,j].reshape(-1,1)

                    if j == minibatch_train_y.shape[1]-1:
                        loss += self.update_parameter(batch_train_x, batch_train_y, learning_rate, clip_value, minibatch_size, update=True)
                    else:
                        loss += self.update_parameter(batch_train_x, batch_train_y, learning_rate, clip_value, minibatch_size, update=False)

                if ep % 10:
                    pred_train_y = self.predict(train_x)
                    train_accuracy = Util.calc_accuracy(train_y, pred_train_y)
                    print(f"Epoch: {ep}, Optimizer: {optimizer}, Loss: {loss/minibatch_train_y.shape[1]}, Train Accuracy: {train_accuracy}")

            else:
                raise ValueError("Invalid optimizer type!")

#%% Test code
if __name__ == "__main__":

    import tensorflow as tf
    from tensorflow.keras import datasets
    from tensorflow.keras.utils import to_categorical

    # Load MNIST dataset
    (train_images, train_labels), (test_images, test_labels) = datasets.mnist.load_data()

    # Reshape and normalize images
    train_images = train_images.reshape((60000, 28 * 28)).astype('float32') / 255
    test_images = test_images.reshape((10000, 28 * 28)).astype('float32') / 255

    # One-hot encode labels
    train_labels_one_hot = to_categorical(train_labels)
    test_labels_one_hot = to_categorical(test_labels)

    # Transpose the data to match the requested shape
    train_x = train_images.T
    train_y = train_labels_one_hot.T
    test_x = test_images.T
    test_y = test_labels_one_hot.T

    network = FFNN("cross_entropy")
    is_train = False

    if is_train:
        network.add_layer(Fully_Connected_Layer(784,64))
        network.add_layer(Activation_Layer(64,"relu"))
        network.add_layer(Fully_Connected_Layer(64,32))
        network.add_layer(Activation_Layer(32,"relu"))
        network.add_layer(Fully_Connected_Layer(32,16))
        network.add_layer(Activation_Layer(16,"relu"))
        network.add_layer(Fully_Connected_Layer(16,10))
        network.add_layer(Activation_Layer(10,"softmax"))

        network.train(train_x, train_y, optimizer="SGD",
            epoch=10000, learning_rate=0.01, clip_value=5, minibatch_size=100)

        network.save_model("FFNN/class_model.pkl")

    else:
        network.load_model("FFNN/class_model.pkl")

    pred_test_y = network.predict(test_x)
    test_accuracy = Util.calc_accuracy(test_y, pred_test_y)
    print(f"Test Accuracy: {test_accuracy}")





