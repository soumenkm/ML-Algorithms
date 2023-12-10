#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec  2 20:03:52 2023

@author: soumensmacbookair
"""

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

train_x = tf.range(10,dtype=tf.float32)
train_y = tf.constant([1.0, 1.3, 3.1, 2.0, 5.0, 6.3, 6.6, 7.4, 8.0, 9.0])
train_x_norm = (train_x -
                tf.math.reduce_mean(train_x))/tf.math.reduce_std(train_x)

val_x = tf.range(10,20,dtype=tf.float32)
val_y = tf.constant([10.0, 10.3, 12.1, 11.0, 14.0, 15.3, 15.6, 16.4, 17.0, 18.0])
val_x_norm = (val_x -
              tf.math.reduce_mean(train_x))/tf.math.reduce_std(train_x)

test_x = tf.linspace(1.5, 10.5, 10)
test_y = tf.constant([1.9, 3.9, 3.0, 6.0, 7.3, 6.1, 5.4, 7.0, 10.0, 12.0])

test_x_norm = (test_x -
               tf.math.reduce_mean(train_x))/tf.math.reduce_std(train_x)

# plt.plot(train_x_norm, train_y, "o--")
# plt.plot(val_x_norm, val_y, "o--")

train_ds = tf.data.Dataset.from_tensor_slices((train_x_norm, train_y))
val_ds = tf.data.Dataset.from_tensor_slices((val_x_norm, val_y))
test_ds = tf.data.Dataset.from_tensor_slices((test_x_norm, test_y))

"""
The original dataset contains 10 examples in the training
batch_size to be used = 2 examples
num_of_batches = 10/2 = 5 = steps_per_epoch
we use the same information for both training and validation
"""

train_ds = train_ds.shuffle(buffer_size=10).repeat().batch(batch_size=2)
val_ds = val_ds.shuffle(buffer_size=10).repeat().batch(batch_size=2)
test_ds = test_ds.batch(batch_size=2)

#%% Subclassing Model
# @tf.keras.utils.register_keras_serializable()
class LinearRegressionModel(tf.keras.Model):

    def __init__(self):
        super(LinearRegressionModel, self).__init__(name="LR-model")
        self.layer0 = tf.keras.layers.InputLayer(input_shape=(None,1),
                                                 name="Input-Layer-Unit-1")
        self.layer1 = tf.keras.layers.Dense(units=1,
                                            activation="linear",
                                            name="Dense-Layer-Unit-1")

    def call(self, inputs):
        a0 = self.layer0(tf.reshape(inputs, (-1,1)))
        a1 = self.layer1(a0)
        return a1

    def summary(self):
        self.layers_list = [self.layer0, self.layer1]
        inputs = tf.keras.Input(shape=(1,),
                                name="Input-Layer-Unit-1")
        x = inputs
        for layer in self.layers_list[1:]:
            outputs = layer(x)
            x = outputs

        self.model_func = tf.keras.Model(inputs, outputs, name="LR-model")
        self.model_func.summary()

model = LinearRegressionModel()
model.build(input_shape=(None,1))
model.summary()

model.compile(optimizer=tf.keras.optimizers.legacy.SGD(learning_rate=0.01,
                                                       momentum=0),
              loss=tf.keras.losses.MeanSquaredError(),
              metrics=[tf.keras.metrics.R2Score()])

early_stopping_callback = tf.keras.callbacks.EarlyStopping(monitor="loss",
                                                           min_delta=0.001,
                                                           patience=3,
                                                           verbose=1,
                                                           mode="min")

tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir="./logs",
                                                      histogram_freq=1,
                                                      write_graph=False,
                                                      write_images=True,
                                                      update_freq="epoch")

filepath = "./checkpoint/weights-epoch-{epoch}.ckpt"
ckpt_callback = tf.keras.callbacks.ModelCheckpoint(filepath=filepath,
                                                   verbose=1,
                                                   save_weights_only=True,
                                                   save_freq=50)

history = model.fit(x=train_ds,
                    epochs=100,
                    verbose=1,
                    validation_data=val_ds,
                    steps_per_epoch=5,
                    validation_steps=5,
                    validation_freq=1,
                    callbacks=[early_stopping_callback,
                               tensorboard_callback,
                               ckpt_callback])

model.save_weights(filepath.format(epoch=history.history["loss"].__len__()))

#%% Evaluate the model
eval_result = model.evaluate(x=test_ds,
                             verbose=1,
                             steps=None)

test_pred = model.predict(test_ds.map(lambda x,y: x),
                          verbose=1,
                          steps=None).reshape(-1,)

plt.plot(test_x_norm, test_y, "o--", label="test_true_data")
plt.plot(test_x_norm, test_pred, "o--", label="test_predicted_data")
plt.legend()

#%% Save and Load weights
# model1 = LinearRegressionModel()
# model1.build(input_shape=(None,1))

# model1.compile(optimizer=tf.keras.optimizers.legacy.SGD(learning_rate=0.01,
#                                                         momentum=0),
#               loss=tf.keras.losses.MeanSquaredError(),
#               metrics=[tf.keras.metrics.R2Score()])

# latest_checkpoint = tf.train.latest_checkpoint("./checkpoint")
# print("Loading latest checkpoint - ", latest_checkpoint)
# model1.load_weights(latest_checkpoint)

# eval_result = model1.evaluate(x=test_ds,
#                               verbose=1,
#                               steps=None)

#%% Save and Load model
# model_filepath = "./model/model-LR"
# model.save(model_filepath)

# model2 = tf.keras.models.load_model(model_filepath)

# # model2 = tf.keras.models.load_model(model_filepath, compile=False)
# # model2.compile(optimizer=tf.keras.optimizers.legacy.SGD(learning_rate=0.01,
# #                                                         momentum=0),
# #                loss=tf.keras.losses.MeanSquaredError(),
# #                metrics=[tf.keras.metrics.R2Score()])

# eval_result = model2.evaluate(x=test_ds,
#                               verbose=1,
#                               steps=None)









