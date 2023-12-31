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

#%% Gradient Tape
x = tf.constant(2.0)

with tf.GradientTape(persistent=False) as tape:
    tape.watch(x)
    f = x ** 2

grad = tape.gradient(target=f, sources=x)

#%% Jacobian
x = tf.constant([[1,2,3]], dtype=tf.float32)
b = tf.Variable([[1,2,3,4]], dtype=tf.float32, trainable=True)
W = tf.Variable([[1,2,3,4],[5,6,7,8],[9,10,11,12]], dtype=tf.float32, trainable=True)

with tf.GradientTape(persistent=False) as tape:
    tape.watch(x)
    y = tf.matmul(x,W) + b

grad = tape.jacobian(target=y, sources=x)
print(grad)

#%% Gradient
x = tf.constant([[1,2,3],[2,3,4],[3,4,5]], dtype=tf.float32)
b = tf.Variable([1,2,3,4], dtype=tf.float32, trainable=True)
W = tf.Variable([[1,2,3,4],[5,6,7,8],[9,10,11,12]], dtype=tf.float32, trainable=True)

with tf.GradientTape(persistent=False) as tape:
    y = tf.matmul(x,W) + b
    L = tf.reduce_sum(y**2, axis=-1)

grad = tape.jacobian(target=L, sources=[W,b])

#%% Model trainable variable
layer = tf.keras.layers.Dense(units=3, activation="linear")
x = tf.ones(shape=(1,2))
with tf.GradientTape() as tape:
    y = layer(x)
    loss = tf.reduce_mean(y**2, axis=-1)

grad = tape.gradient(loss, layer.trainable_variables)
print(grad)

#%% Intermediate gradient
x = tf.constant(2.0)
with tf.GradientTape() as tape:
    tape.watch(x)
    y1 = x ** 2
    y2 = y1 ** 2
    loss = y2 ** 2

grad = tape.gradient(loss, [y2, y1, x])
print(grad)

#%% Gradient on multiple target
x = tf.Variable(2.0, trainable=True)
with tf.GradientTape(persistent=True) as tape:
    y0 = x ** 2
    y1 = 1/x

grad = tape.gradient([y0, y1], x)
print(grad)

# grad1 = tape.gradient(y0, x)
# grad2 = tape.gradient(y1, x)
# print(grad1, grad2)

#%% Sum of gradient
x = tf.linspace(0, 10, 100)
with tf.GradientTape(persistent=True) as tape:
    tape.watch(x)
    y = tf.math.sin(x)

grad = tape.gradient(y, x)
plt.plot(x, grad, label="dy/dx=cos(x)")
plt.plot(x, y, label="y=sin(x)")
plt.legend()

#%% Different gradient
x1 = tf.constant([1,2,3.0])
x2 = tf.constant([4,5,6.0])
with tf.GradientTape(persistent=True) as tape:
    tape.watch([x1, x2])
    y = tf.math.pow(x1,2) + tf.math.pow(x2,2)

grad = tape.gradient(y, [x1, x2])
print(grad)

#%% Subgradient
x = tf.constant(2.0)
with tf.GradientTape() as tape:
    tape.watch(x)
    if x <= 0.0:
        y = x * 0
    else:
        y = x

grad = tape.gradient(y, x)
print(grad)

#%% Case 1 of None
x = tf.Variable(2.0, trainable=True)
for epoch in range(5):
    with tf.GradientTape() as tape:
        y = x ** 2

    grad = tape.gradient(y, x)
    print(f"Epoch: {epoch}, Grad: {grad}, x: ", x)
    x = x + 1 # x.assign_add(1) to get correct gradient

#%% Weird case
x = tf.Variable(2.0, trainable=True)
with tf.GradientTape() as tape:
    x = x + 1
    y = x ** 2

grad = tape.gradient(y, x)
print(x, grad)

x = tf.Variable(2.0, trainable=True)
with tf.GradientTape() as tape:
    y = x ** 2
    x = x + 1

grad = tape.gradient(y, x)
print(x, grad)

#%% Case 3 of None
x = tf.Variable([[1.0, 2.0],
                 [3.0, 4.0]], dtype=tf.float32)

with tf.GradientTape() as tape:
  y = np.mean(x2, axis=0)
  y = tf.reduce_mean(y, axis=0)

print(tape.gradient(y, x))

#%% Case 4 of None
x = tf.Variable([1,2,3])

with tf.GradientTape() as tape:
  y = x ** 2

print(tape.gradient(y, x))

#%% State of variable
x0 = tf.Variable(2.0)
x1 = tf.Variable(3.0)
with tf.GradientTape(persistent=True) as tape:
    x1.assign_add(x0)
    y = x1 ** 2

print(tape.gradient(y, x0))
print(tape.gradient(y, x1))

#%% Finals







