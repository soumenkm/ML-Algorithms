#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 10 20:03:52 2023

@author: soumensmacbookair
"""

#%% Import the libraries
import tensorflow as tf
import tensorflow_datasets as tfds

# Import the dataset
ds, info = tfds.load('mnist', with_info=True)
train_val_ds_orig = ds["train"]
test_ds_orig = ds["test"]

tfds.show_examples(train_val_ds_orig, info)

buffer_size = 1000
train_val_ds_orig = train_val_ds_orig.shuffle(buffer_size,
                                              reshuffle_each_iteration=False)
train_ds_orig = train_val_ds_orig.take(int(train_val_ds_orig.__len__().numpy() * 0.9))
val_ds_orig = train_val_ds_orig.skip(int(train_val_ds_orig.__len__().numpy() * 0.9))

def preprocess_data(data):
    """data must be a dictionary"""

    x = tf.cast(data["image"], tf.float32)
    x = x/255.0
    y = tf.cast(data["label"], tf.int64)
    y = tf.one_hot(y, depth=10)

    return (tf.reshape(x, (784,)),
            tf.reshape(y, (10,)))

train_ds = train_ds_orig.map(preprocess_data)
val_ds = val_ds_orig.map(preprocess_data)
test_ds = test_ds_orig.map(preprocess_data)

batch_size = 100
train_data_size = train_ds.__len__()
val_data_size = val_ds.__len__()
test_data_size = test_ds.__len__()

num_of_train_batch = int(train_data_size / batch_size)
num_of_val_batch = int(val_data_size / batch_size)
num_of_test_batch = int(test_data_size / batch_size)

train_ds = train_ds.shuffle(buffer_size).repeat(count=None).batch(batch_size)
val_ds = val_ds.shuffle(buffer_size).repeat(count=None).batch(batch_size)
test_ds = test_ds.batch(batch_size)

# Build and compile the model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(units=64, activation="relu", kernel_regularizer="l2", bias_regularizer="l2", name="FC-layer-units-64", input_shape=(784,)),
    tf.keras.layers.Dense(units=32, activation="relu", kernel_regularizer="l2", bias_regularizer="l2", name="FC-layer-units-32"),
    tf.keras.layers.Dense(units=16, activation="relu", kernel_regularizer="l2", bias_regularizer="l2", name="FC-layer-units-16"),
    tf.keras.layers.Dense(units=10, activation="softmax", kernel_regularizer="l2", bias_regularizer="l2", name="Output-layer-units-10"),
    ], name="FFNN-model")

model.summary()
model.compile(optimizer=tf.keras.optimizers.legacy.SGD(learning_rate=0.01,
                                                       momentum=0.1,
                                                       clipvalue=None),
              loss=tf.keras.losses.CategoricalCrossentropy(from_logits=False),
              metrics=[tf.keras.metrics.CategoricalAccuracy(),
                       tf.keras.metrics.F1Score()])

# Train the model
tb_callback = tf.keras.callbacks.TensorBoard(log_dir="./logs",
                                             histogram_freq=1,
                                             write_graph=True,
                                             write_images=True,
                                             update_freq="epoch")

es_callback = tf.keras.callbacks.EarlyStopping(monitor="loss",
                                               min_delta=0.001,
                                               patience=5,
                                               verbose=1,
                                               mode="min")

ckpt_callback = tf.keras.callbacks.ModelCheckpoint(filepath="./checkpoint/weights-epoch-{epoch}.ckpt",
                                                   verbose=1,
                                                   save_weights_only=True,
                                                   save_freq=10*num_of_train_batch)

model_history = model.fit(x=train_ds,
                          epochs=100,
                          verbose=1,
                          validation_data=val_ds,
                          steps_per_epoch=num_of_train_batch,
                          validation_steps=num_of_val_batch,
                          validation_freq=10,
                          callbacks=[tb_callback,
                                     es_callback,
                                     ckpt_callback])

model.save_weights(f"./checkpoint/weights-epoch-{model_history.history['loss'].__len__()}.ckpt")

# Testing the model
test_eval_result = model.evaluate(x=test_ds,
                                  verbose=1,
                                  steps=num_of_test_batch)

test_pred_result = model.predict(x=test_ds.map(lambda x,y: x),
                                 verbose=1,
                                 steps=num_of_test_batch)

test_pred_result = test_pred_result.argmax(axis=1)

# Saving the model
model.save("./model/model-FFNN")

# Load the saved model
model_new = tf.keras.models.load_model("./model/model-FFNN", compile=True)
test_eval_result = model_new.evaluate(x=test_ds,
                                      verbose=1,
                                      steps=num_of_test_batch)





