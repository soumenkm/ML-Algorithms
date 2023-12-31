#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 11 11:57:46 2023

@author: soumensmacbookair
"""

#%% Import the libraries
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

#%% Import and preprocess the dataset
ds, ds_info = tfds.load("iris", with_info=True)
ds = ds["train"]
dataset_size = ds.__len__()

ds = ds.shuffle(buffer_size=dataset_size, reshuffle_each_iteration=False)

train_ds_orig = ds.take(100)
val_ds_orig = ds.skip(100).take(20)
test_ds_orig = ds.skip(120)

train_ds_size = train_ds_orig.__len__()
val_ds_size = val_ds_orig.__len__()
test_ds_size = test_ds_orig.__len__()

batch_size = 10
num_of_train_batch = int(train_ds_size/batch_size)
num_of_val_batch = int(val_ds_size/batch_size)
num_of_test_batch = int(test_ds_size/batch_size)

def preprocess_data(data):
    """data must be a dictionary of two keys - features and label"""

    x1 = tf.math.sqrt(data["features"][0] ** 2 + data["features"][2] ** 2) # length
    x2 = tf.math.sqrt(data["features"][1] ** 2 + data["features"][3] ** 2) # width
    x = tf.stack([x1,x2], axis=0)
    y = data["label"]
    y = tf.one_hot(y, depth=3, axis=0)

    return (x, y)

train_ds = train_ds_orig.map(preprocess_data)
val_ds = val_ds_orig.map(preprocess_data)
test_ds = test_ds_orig.map(preprocess_data)

train_ds = train_ds.shuffle(train_ds_size).repeat().batch(batch_size)
val_ds = val_ds.shuffle(val_ds_size).repeat().batch(batch_size)
test_ds = test_ds.batch(batch_size)

# Build and compile the model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(units=3,
                          kernel_initializer="glorot_normal",
                          kernel_regularizer="l2",
                          bias_regularizer="l2",
                          activation="softmax",
                          name="Output-layer-units-3",
                          input_shape=(2,))
    ], name="LogReg-model")

model.summary()
model.compile(optimizer=tf.keras.optimizers.legacy.SGD(learning_rate=0.01,
                                                       momentum=0.1),
              loss=tf.keras.losses.CategoricalCrossentropy(from_logits=False),
              metrics=[tf.keras.metrics.CategoricalAccuracy(),
                       tf.keras.metrics.F1Score()])

#%% Train the model
tb_callback = tf.keras.callbacks.TensorBoard(log_dir="./logs",
                                             histogram_freq=1,
                                             write_graph=False,
                                             write_images=True,
                                             update_freq="epoch")

ckpt_callback = tf.keras.callbacks.ModelCheckpoint(filepath="./checkpoint/weights-epoch-{epoch}.ckpt",
                                                   verbose=0,
                                                   save_weights_only=True,
                                                   save_freq=5*num_of_train_batch)

num_of_epoch = 500
model_history = model.fit(x=train_ds,
                          epochs=num_of_epoch,
                          verbose=1,
                          validation_data=val_ds,
                          steps_per_epoch=num_of_train_batch,
                          validation_steps=num_of_val_batch,
                          validation_freq=1,
                          callbacks=[tb_callback, ckpt_callback])

history = model_history.history

#%% Load the best model and Test the model
latest_ckpt = tf.train.latest_checkpoint("./checkpoint")
model.load_weights(latest_ckpt)

pred_result = model.predict(x=test_ds.map(lambda x,y: x),
                            verbose=1,
                            steps=None).argmax(axis=1)

eval_result = model.evaluate(x=test_ds,
                             verbose=1,
                             steps=None)

#%% Save the model
model.save("./model/LogReg-model")

#%% Visualize the training dataset and decision boundary
vis_ds = train_ds.take(10).unbatch()
vis_ds_size = batch_size * 10
x = np.zeros(shape=(vis_ds_size, 2))
y = np.zeros(shape=(vis_ds_size,))

for i,batch in enumerate(vis_ds):
    x[i,:] = batch[0].numpy()
    y[i] = tf.argmax(batch[1], axis=0).numpy()

x1 = np.linspace(x[:,0].min() - 1, x[:,0].max() + 1, 100)
x2 = np.linspace(x[:,1].min() - 1, x[:,1].max() + 1, 100)

mesh_x1, mesh_x2 = np.meshgrid(x1, x2)
mesh = np.stack([mesh_x1, mesh_x2], axis=0)
mesh_data = mesh.reshape(2, x1.shape[0]**2).T
mesh_data = tf.data.Dataset.from_tensor_slices(mesh_data)
mesh_data = mesh_data.batch(batch_size)

def predict_on_mesh(model: tf.keras.Sequential, epoch):

    model.load_weights(f"./checkpoint/weights-epoch-{epoch}.ckpt")
    pred_mesh_data = model.predict(x=mesh_data,
                                   verbose=0,
                                   steps=None).argmax(axis=1)
    return pred_mesh_data

fig, ax = plt.subplots(figsize=(10, 6))

def update_frame(epoch):

    if (epoch+1) % 5 != 0:
        return

    ax.clear()

    ax.scatter(x=x[:,0][y==0], y=x[:,1][y==0], c="purple", marker="o", label="class 0")
    ax.scatter(x=x[:,0][y==1], y=x[:,1][y==1], c="green", marker="x", label="class 1")
    ax.scatter(x=x[:,0][y==2], y=x[:,1][y==2], c="blue", marker="d", label="class 2")

    pred_mesh_data = predict_on_mesh(model, epoch=epoch+1)

    ax.contourf(mesh_x1, mesh_x2, pred_mesh_data.reshape(mesh_x1.shape),
                alpha=0.5, levels=[-0.5, 0.5, 1.5, 2.5],
                colors=["purple", "green", "blue"])

    ax.set_xlabel('Feature length')
    ax.set_ylabel('Feature width')
    ax.set_title(f'Decision boundary at epoch {epoch+1}')
    ax.legend()

animation = FuncAnimation(fig, update_frame, frames=num_of_epoch, interval=100,
                          blit=False, repeat=False)
animation.save('./model/decision_boundary_LogReg_model.gif', writer='pillow')

