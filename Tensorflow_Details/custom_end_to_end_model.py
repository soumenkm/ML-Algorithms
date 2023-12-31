#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 31 11:57:46 2023

@author: soumensmacbookair
"""

#%% Import the libraries
import time, tqdm
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt

#%% Load the dataset
def get_dataset():

    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    x_train = x_train.reshape((-1,784))
    x_test = x_test.reshape((-1,784))/255.0
    y_train = y_train.reshape((-1,1))
    y_test = y_test.reshape((-1,1))

    x_val = x_train[:10000,:]/255.0
    y_val = y_train[:10000,:]
    x_train = x_train[10000:,:]/255.0
    y_train = y_train[10000:,:]

    x_train = tf.cast(x_train, tf.float32)
    x_test = tf.cast(x_test, tf.float32)
    x_val = tf.cast(x_val, tf.float32)
    y_train = tf.cast(y_train, tf.int64)
    y_test = tf.cast(y_test, tf.int64)
    y_val = tf.cast(y_val, tf.int64)

    # Prepare the dataset pipeline
    train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test))
    val_ds = tf.data.Dataset.from_tensor_slices((x_val, y_val))

    train_ds = train_ds.shuffle(1024).batch(64)
    test_ds = test_ds.batch(64)
    val_ds = val_ds.batch(64)

    train_ds = train_ds.map(lambda x, y: (tf.cast(x, tf.float32), tf.cast(y, tf.int64)))
    val_ds = val_ds.map(lambda x, y: (tf.cast(x, tf.float32), tf.cast(y, tf.int64)))
    test_ds = test_ds.map(lambda x, y: (tf.cast(x, tf.float32), tf.cast(y, tf.int64)))

    return train_ds, val_ds, test_ds

#%% Define the Model
class DenseLayer(tf.keras.layers.Layer):

    def __init__(self, n_out: int, activation: str="linear", **kwargs):

        super(DenseLayer, self).__init__(**kwargs)
        self.n_out = n_out
        self.act = activation
        self.W = None
        self.b = None

    def build(self, input_shape: tf.Tensor):
        """
        input_shape.shape must be (batch_size, num_units)
        """

        assert len(input_shape) == 2, "input_shape.shape must be (batch_size, num_units)"
        self.n_in = input_shape[-1]

        self.W = self.add_weight(name="weight",
                                 shape=(self.n_in, self.n_out),
                                 initializer="he_normal",
                                 trainable=True,
                                 dtype=tf.float32) # (n_in, n_out)

        self.b = self.add_weight(name="bias",
                                 shape=(self.n_out,),
                                 initializer="zeros",
                                 trainable=True,
                                 dtype=tf.float32) # (n_out,)

    def call(self, inputs: tf.Tensor):
        """
        inputs.shape must be (batch_size, 784)
        outputs.shape must be (batch_size, self.n_out)
        """

        inputs = tf.cast(inputs, dtype=tf.float32)

        assert inputs.shape.rank == 2, "inputs.shape must be (batch_size, num_units)"

        outputs = tf.matmul(inputs, self.W) + self.b # (batch_size, n_out)

        if self.act == "linear":
            outputs = outputs * 1.0
        elif self.act == "relu":
            outputs = tf.nn.relu(outputs) # (batch_size, n_out)
        elif self.act == "softmax":
            outputs = tf.nn.softmax(outputs, axis=-1) # (batch_size, n_out)
        else:
            raise NotImplementedError("invalid activation")

        tf.debugging.assert_shapes([(outputs, (None, self.n_out))], message="outputs.shape must be (batch_size, self.n_out)")

        return outputs

class FFNNModel(tf.keras.Model):

    def __init__(self):

        super(FFNNModel, self).__init__(name="FFNN-model")

        self.layer1 = DenseLayer(n_out=64, activation="relu", name="Dense-layer-1")
        self.layer2 = DenseLayer(n_out=32, activation="relu", name="Dense-layer-2")
        self.layer3 = DenseLayer(n_out=16, activation="relu", name="Dense-layer-3")
        self.layer4 = DenseLayer(n_out=10, activation="softmax", name="Output-layer")

    def call(self, inputs: tf.Tensor):
        """
        inputs.shape must be (batch_size, 784)
        outputs.shape must be (batch_size, 10)
        """

        inputs = tf.cast(inputs, dtype=tf.float32)

        tf.debugging.assert_shapes([(inputs, (None, 784))], message="inputs.shape must be (batch_size, 784)")

        for layer in [self.layer1, self.layer2, self.layer3, self.layer4]:
            outputs = layer(inputs) # at last it should be (batch_size, 10)
            inputs = outputs

        tf.debugging.assert_shapes([(outputs, (None, 10))], message="outputs.shape must be (batch_size, 10)")

        return outputs

#%% Define the loss function
def loss_function(y_true: tf.Tensor, y_pred: tf.Tensor):
    """
    y_true.shape must be (batch_size, 1)
    y_pred.shape must be (batch_size, 10)
    output loss.shape must be ()
    """

    y_true = tf.cast(y_true, dtype=tf.int64)
    y_pred = tf.cast(y_pred, dtype=tf.float32)

    tf.debugging.assert_shapes(shapes=[(y_true, (None,1))], message="y_true.shape must be (batch_size, 1)")
    tf.debugging.assert_shapes(shapes=[(y_pred, (None,10))], message="y_pred.shape must be (batch_size, 10)")
    tf.debugging.assert_near(tf.reduce_sum(y_pred, axis=-1), 1.0, atol=0.1, message="y_pred must be softmax probabilities")

    y_true = tf.squeeze(y_true) # (batch_size,)
    y_true_oh = tf.one_hot(y_true, y_pred.shape[-1]) # (batch_size, 10)

    loss = -1 * tf.reduce_sum(y_true_oh * tf.math.log(y_pred), axis=-1) # (batch_size,)
    loss = tf.reduce_mean(loss, axis=-1) # mean over batch size, just a scalar ()

    tf.debugging.assert_shapes(shapes=[(loss, ())], message="loss.shape must be ()")

    return loss

#%% Define metric function
def metric_function(y_true: tf.Tensor, y_pred: tf.Tensor):
    """
    y_true.shape must be (batch_size, 1)
    y_pred.shape must be (batch_size, 10)
    output acc.shape must be ()
    """

    y_true = tf.cast(y_true, dtype=tf.int64)
    y_pred = tf.cast(y_pred, dtype=tf.float32)

    tf.debugging.assert_shapes(shapes=[(y_true, (None,1))], message="y_true.shape must be (batch_size,1)")
    tf.debugging.assert_shapes(shapes=[(y_pred, (None,10))], message="y_pred.shape must be (batch_size,10)")
    tf.debugging.assert_near(tf.reduce_sum(y_pred, axis=-1), 1.0, atol=0.0001, message="y_pred must be softmax probabilities")

    y_true = tf.squeeze(y_true) # (batch_size,)
    y_pred_label = tf.argmax(y_pred, axis=-1, output_type=tf.int64) # (batch_size,)

    issame = tf.cast(y_true == y_pred_label, tf.int32)
    acc = tf.cast(tf.reduce_sum(issame, axis=-1) / tf.size(issame), tf.float32) # ()

    tf.debugging.assert_shapes(shapes=[(acc, ())], message="acc.shape must be ()")

    return acc

#%% Calculate the gradient of parameters
def calculate_gradient(model: tf.keras.Model, input_x: tf.Tensor, output_y_true: tf.Tensor):
    """
    input_x.shape must be (batch_size, 784)
    output_y_true.shape must be (batch_size, 1)
    outputs acc.shape and loss.shape must be (), grads must be a collection of gradients
    """

    input_x = tf.cast(input_x, dtype=tf.float32)
    output_y_true = tf.cast(output_y_true, dtype=tf.float32)

    tf.debugging.assert_shapes(shapes=[(output_y_true, (None,1))], message="output_y_true.shape must be (batch_size, 1)")
    tf.debugging.assert_shapes(shapes=[(input_x, (None,784))], message="input_x.shape must be (batch_size, 784)")
    assert isinstance(model, tf.keras.Model), "model must be a valid keras model"

    with tf.GradientTape() as tape:
        y_pred = model(input_x, training=True) # (batch_size, 10)
        loss = loss_function(output_y_true, y_pred) # ()

    tf.debugging.assert_shapes(shapes=[(loss, ())], message="loss.shape must be ()")
    acc = metric_function(output_y_true, y_pred) # ()
    tf.debugging.assert_shapes(shapes=[(acc, ())], message="acc.shape must be ()")

    grads = tape.gradient(loss, model.trainable_variables)

    return loss, acc, grads

#%% Custom training loop
def fit(model: tf.keras.Model,
        train_dataset: tf.data.Dataset,
        val_dataset: tf.data.Dataset,
        epochs: int, learning_rate: float):

    optimizer = tf.keras.optimizers.legacy.SGD(learning_rate=0.01)
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    for ep in range(epochs):

        x_train_list = []
        y_train_list = []
        x_val_list = []
        y_val_list = []

        for x_batch, y_batch in train_dataset:

            _, _, grads = calculate_gradient(model, x_batch, y_batch)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))

            x_train_list.append(x_batch)
            y_train_list.append(y_batch)

        for x_batch, y_batch in val_dataset:

            x_val_list.append(x_batch)
            y_val_list.append(y_batch)

        x_train_tensor = tf.concat(x_train_list, axis=0)
        y_train_tensor = tf.concat(y_train_list, axis=0)
        x_val_tensor = tf.concat(x_val_list, axis=0)
        y_val_tensor = tf.concat(y_val_list, axis=0)

        y_val_pred = model(x_val_tensor, training=False)
        y_train_pred = model(x_train_tensor, training=False)

        val_loss = loss_function(y_val_tensor, y_val_pred)
        val_acc = metric_function(y_val_tensor, y_val_pred)

        train_loss = loss_function(y_train_tensor, y_train_pred)
        train_acc = metric_function(y_train_tensor, y_train_pred)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        print(f"epoch {ep+1}/{epochs}, train loss: {train_loss:.2}, train acc: {train_acc:.2}, Val loss: {val_loss:.2}, Val acc: {val_acc:.2}")

    return history

#%% Testing loop
def evaluate(model: tf.keras.Model, test_dataset: tf.data.Dataset):

    x_test_list = []
    y_test_list = []

    for x_batch, y_batch in test_dataset:

        x_test_list.append(x_batch)
        y_test_list.append(y_batch)

    x_test_tensor = tf.concat(x_test_list, axis=0)
    y_test_tensor = tf.concat(y_test_list, axis=0)

    y_test_pred = model(x_test_tensor, training=False)

    test_loss = loss_function(y_test_tensor, y_test_pred)
    test_acc = metric_function(y_test_tensor, y_test_pred)

    print(f"test loss: {test_loss:.2}, test acc: {test_acc:.2}")

    return {"test_loss": test_loss, "test_acc": test_acc}

#%% Main code
if __name__ == "__main__":

    model = FFNNModel()
    model.build(input_shape=(None, 784))
    model.summary()

    train_ds, val_ds, test_ds = get_dataset()

    history = fit(model, train_ds, val_ds, epochs=5, learning_rate=0.01)
    evaluate(model, test_ds)





