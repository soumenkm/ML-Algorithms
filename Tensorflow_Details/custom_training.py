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

#%% Prepare the dataset pipeline
train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train))
test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test))
val_ds = tf.data.Dataset.from_tensor_slices((x_val, y_val))

train_ds = train_ds.shuffle(1024).batch(64)
test_ds = test_ds.batch(64)
val_ds = val_ds.batch(64)

train_ds = train_ds.map(lambda x, y: (tf.cast(x, tf.float32), tf.cast(y, tf.int64)))
val_ds = val_ds.map(lambda x, y: (tf.cast(x, tf.float32), tf.cast(y, tf.int64)))
test_ds = test_ds.map(lambda x, y: (tf.cast(x, tf.float32), tf.cast(y, tf.int64)))

#%% Define the model
model = tf.keras.Sequential([tf.keras.layers.Dense(units=64, activation="relu", kernel_initializer="he_normal"),
                             tf.keras.layers.Dense(units=32, activation="relu", kernel_initializer="he_normal"),
                             tf.keras.layers.Dense(units=16, activation="relu", kernel_initializer="he_normal"),
                             tf.keras.layers.Dense(units=10, activation="softmax", kernel_initializer="glorot_normal")])

model.build(input_shape=(None,784))
model.summary()

#%% Define custom loss function
def loss_function(y_true, y_pred):

    tf.debugging.assert_shapes(shapes=[(y_true, (None,1))], message="y_true.shape must be (batch_size,1)")
    tf.debugging.assert_shapes(shapes=[(y_pred, (None,10))], message="y_pred.shape must be (batch_size,10)")
    tf.debugging.assert_near(tf.reduce_sum(y_pred, axis=-1), 1.0, atol=0.0001, message="y_pred must be softmax probabilities")

    y_true = tf.squeeze(y_true) # (None,)
    y_true = tf.cast(y_true, tf.int64)

    y_true_oh = tf.one_hot(y_true, y_pred.shape[-1]) # (None, 10)
    loss = -1 * tf.reduce_sum(y_true_oh * tf.math.log(y_pred), axis=-1) # (None,)
    loss = tf.reduce_mean(loss, axis=-1) # mean over batch size, just a scalar ()

    # loss_obj = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)
    # loss_tf = loss_obj(y_true, y_pred)

    # tf.debugging.assert_near(loss, loss_tf, atol=0.001, message="loss must be same as loss by tf")

    return loss

#%% Define custom metric function
def metric_function(y_true, y_pred):

    tf.debugging.assert_shapes(shapes=[(y_true, (None,1))], message="y_true.shape must be (batch_size,1)")
    tf.debugging.assert_shapes(shapes=[(y_pred, (None,10))], message="y_pred.shape must be (batch_size,10)")
    tf.debugging.assert_near(tf.reduce_sum(y_pred, axis=-1), 1.0, atol=0.0001, message="y_pred must be softmax probabilities")

    y_true = tf.squeeze(y_true) # (None,)
    y_true = tf.cast(y_true, tf.int64)
    y_pred_label = tf.argmax(y_pred, axis=-1, output_type=tf.int64) # (None,)

    issame = tf.cast(y_true == y_pred_label, tf.int32)
    acc = tf.cast(tf.reduce_sum(issame, axis=-1) / tf.size(issame), tf.float32) # ()

    # acc_obj = tf.keras.metrics.SparseCategoricalAccuracy()
    # acc_tf = acc_obj(y_true, y_pred)

    # # print(acc, acc_tf)
    # tf.debugging.assert_near(acc, acc_tf, atol=0.001, message="acc must be same as acc by tf")

    return acc

#%% Train the model
# model.compile("sgd", loss_function, [metric_function, "sparse_categorical_accuracy"])
# model.fit(train_ds, epochs=10, verbose=1, validation_data=val_ds)

#%% Calculate the gradient of parameters
def calculate_gradient(model, input_x, output_y_true):

    tf.debugging.assert_shapes(shapes=[(output_y_true, (None,1))], message="output_y_true.shape must be (batch_size,1)")
    tf.debugging.assert_shapes(shapes=[(input_x, (None,784))], message="input_x.shape must be (batch_size,784)")
    assert isinstance(model, tf.keras.Model), "model must be a valid keras model"

    with tf.GradientTape() as tape:
        y_pred = model(input_x, training=True)
        loss = loss_function(output_y_true, y_pred)

    acc = metric_function(output_y_true, y_pred)
    grads = tape.gradient(loss, model.trainable_variables)

    return loss, acc, grads

#%% Custom training loop
epochs = 20
batch_size = 64
steps = int(x_train.shape[0]/batch_size) + 1
optimizer = tf.keras.optimizers.legacy.SGD(learning_rate=0.01)

for ep in range(epochs):
    t1 = time.time()
    for step in tqdm.tqdm(range(steps), desc=f"epoch: {ep+1}/{epochs}"):
        x_batch = x_train[batch_size*step:batch_size*(step+1),:]
        y_batch = y_train[batch_size*step:batch_size*(step+1),:]

        _, _, grads = calculate_gradient(model, x_batch, y_batch)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))

    y_val_pred = model(x_val, training=False)
    y_train_pred = model(x_train, training=False)

    val_loss = loss_function(y_val, y_val_pred)
    val_acc = metric_function(y_val, y_val_pred)

    train_loss = loss_function(y_train, y_train_pred)
    train_acc = metric_function(y_train, y_train_pred)

    t2 = time.time()
    dt = round(t2-t1, 2)
    print("")
    print(f"epoch {ep+1}/{epochs} completed in {dt} secs, train loss: {train_loss:.2}, train acc: {train_acc:.2}, Val loss: {val_loss:.2}, Val acc: {val_acc:.2}")

#%% Custom training loop with storage in metrics class
epochs = 5
batch_size = 64
steps = int(x_train.shape[0]/batch_size) + 1
optimizer = tf.keras.optimizers.legacy.SGD(learning_rate=0.01)

# Add the state in metrics
train_loss_res = []
train_acc_res = []
time_step_res = []

train_loss_ep = tf.keras.metrics.Mean()
train_acc_ep = tf.keras.metrics.Mean()
time_step_ep = -1

for ep in range(epochs):
    t1 = time.time()
    for step in tqdm.tqdm(range(steps), desc=f"epoch: {ep+1}/{epochs}"):
        x_batch = x_train[batch_size*step:batch_size*(step+1),:]
        y_batch = y_train[batch_size*step:batch_size*(step+1),:]

        loss, acc, grads = calculate_gradient(model, x_batch, y_batch)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))

        train_loss_ep.update_state(loss)
        train_acc_ep.update_state(acc)
        time_step_ep += 1

        train_loss_res.append(train_loss_ep.result().numpy())
        train_acc_res.append(train_acc_ep.result().numpy())
        time_step_res.append(time_step_ep)

    y_train_pred = model(x_train, training=False)
    train_loss = loss_function(y_train, y_train_pred)
    train_acc = metric_function(y_train, y_train_pred)

    t2 = time.time()
    dt = round(t2-t1, 2)

    print("")
    print(f"epoch {ep+1}/{epochs} completed in {dt} secs, train loss: {train_loss:.2}, train acc: {train_acc:.2}")

plt.plot(time_step_res, train_loss_res, label="loss")
plt.plot(time_step_res, train_acc_res, label="accuracy")
plt.legend()


