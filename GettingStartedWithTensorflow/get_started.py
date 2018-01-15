# -*- coding: utf-8 -*-
"""
Created on Mon Jan 15 15:14:01 2018

@author: LeeJiangLee
"""

import tensorflow as tf

#Model parameters
W = tf.Variable([.3], dtype=tf.float32)
b = tf.Variable([-.3], dtype=tf.float32)
#Model input and output
x = tf.placeholder(tf.float32)
linear_model = W*x + b
y = tf.placeholder(tf.float32)

#loss
loss = tf.reduce_sum(tf.square(linear_model-y)) #sum of the squares
#optimizer
