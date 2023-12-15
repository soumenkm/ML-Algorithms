#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 12 11:57:46 2023

@author: soumensmacbookair
"""

# Import the libraries
import time
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt

#%% Graph execution
@tf.function
def func(a,b):
    return a+b

# Create a file writer
logdir = "logs/" + time.strftime("%Y-%m-%d_%H:%M:%S")
file_writer = tf.summary.create_file_writer(logdir)

# Starts the trace
tf.summary.trace_on(graph=True, profiler=False)

# Call the function
a = tf.constant([[2.0, 3.0]])
b = tf.constant([4.0])
c = func(a,b)

# Exports the trace
with file_writer.as_default():
    tf.summary.trace_export(name="function trace", step=0)

