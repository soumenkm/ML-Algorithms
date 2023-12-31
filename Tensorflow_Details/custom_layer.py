#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 30 11:57:46 2023

@author: soumensmacbookair
"""

#%% Import the libraries
import time
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt

#%% Immediate variable creation
class CustomLinearLayer(tf.keras.layers.Layer):

    def __init__(self, num_input_units, num_output_units):

        super(CustomLinearLayer, self).__init__()

        self.num_input_units = num_input_units
        self.num_output_units = num_output_units

        self.W = self.add_weight(name="projection_weight",
                                 shape=(self.num_input_units, self.num_output_units),
                                 dtype=tf.float32,
                                 initializer="glorot_normal",
                                 trainable=True)

        self.b = self.add_weight(name="projection_bias",
                                 shape=(self.num_output_units,),
                                 dtype=tf.float32,
                                 initializer="zeros",
                                 trainable=True)

        self.ones = self.add_weight(name="mask-all-with-ones",
                                    shape=(self.num_output_units,),
                                    dtype=tf.float32,
                                    initializer="ones",
                                    trainable=False)

    def call(self, inputs):

        assert tf.is_tensor(inputs), "inputs must be a tensor"
        assert inputs.shape[-1] == self.num_input_units, "input dimension is incompatible"
        assert inputs.shape.rank == 2, "batch dimension must be present"

        y = tf.matmul(inputs, self.W) + self.b # broadcasted

        return y

#%% Late variable creation
class LateCustomLinearLayer(tf.keras.layers.Layer):

    def __init__(self, num_output_units):

        super(LateCustomLinearLayer, self).__init__()

        self.num_output_units = num_output_units

    def build(self, input_shape):

        self.num_input_units = input_shape[-1]

        self.W = self.add_weight(name="projection_weight",
                                 shape=(self.num_input_units, self.num_output_units),
                                 dtype=tf.float32,
                                 initializer="glorot_normal",
                                 trainable=True)

        self.b = self.add_weight(name="projection_bias",
                                 shape=(self.num_output_units,),
                                 dtype=tf.float32,
                                 initializer="zeros",
                                 trainable=True)

    def call(self, inputs):

        assert tf.is_tensor(inputs), "inputs must be a tensor"
        assert inputs.shape.rank == 2, "batch dimension must be present"

        y = tf.matmul(inputs, self.W) + self.b # broadcasted

        return y

# layer = LateCustomLinearLayer(3) # W should be n by 3 tensor
# # layer.W # it will be error as variable not created
# a = tf.random.uniform([2])
# layer(a) # will throw error but W is created

#%% Build custom MLP layer
class LinearLayer(tf.keras.layers.Layer):

    def __init__(self, num_output_units, **kwargs):

        super(LinearLayer, self).__init__(**kwargs)

        self.num_output_units = num_output_units

    def build(self, input_shape):

        self.num_input_units = input_shape[-1]

        self.W = self.add_weight(name="projection_weight",
                                 shape=(self.num_input_units, self.num_output_units),
                                 dtype=tf.float32,
                                 initializer="glorot_normal",
                                 trainable=True)

        self.b = self.add_weight(name="projection_bias",
                                 shape=(self.num_output_units,),
                                 dtype=tf.float32,
                                 initializer="zeros",
                                 trainable=True)

    def call(self, inputs):

        assert tf.is_tensor(inputs), "inputs must be a tensor"
        assert inputs.shape.rank == 2, "batch dimension must be present"

        y = tf.matmul(inputs, self.W) + self.b # broadcasted

        return y

class MLPLayer(tf.keras.layers.Layer):

    def __init__(self):

        super(MLPLayer, self).__init__(name="MLP-layer")

        self.linear1 = LinearLayer(num_output_units=8, name="linear-1")
        self.linear2 = LinearLayer(num_output_units=4, name="linear-2")
        self.linear3 = LinearLayer(num_output_units=2, name="linear-3")
        self.linear4 = LinearLayer(num_output_units=1, name="linear-4")

    def call(self, inputs):

        assert tf.is_tensor(inputs), "inputs must be a tensor"
        assert inputs.shape.rank == 2, "batch dimension must be present"

        for linear in [self.linear1, self.linear2, self.linear3]:
            outputs = linear(inputs=inputs)
            outputs = tf.nn.relu(outputs)
            inputs = outputs

        outputs = self.linear4(inputs=inputs)

        return outputs

# mlp = MLPLayer()
# # mlp.variables # []
# # mlp.linear1.variables # []

#%% Build custom model
class MLPModel(tf.keras.Model):

    def __init__(self):

        super(MLPModel, self).__init__(name="MLP-model")
        self.mlp_layer = MLPLayer()

    def call(self, inputs):

        outputs = self.mlp_layer(inputs)

        return outputs

# a = tf.random.uniform(shape=(4,2))
# mlp = MLPModel()
# mlp(a)
















