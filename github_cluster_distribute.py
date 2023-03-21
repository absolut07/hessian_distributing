#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 26 19:27:47 2020

This code is also trying to ditribute calculations and update variables.
But here the goal was distributing on a cluster. It was a tough task,
a lot of conflicts between old and new version of TF (1 and 2)

"""
import distributed_nn_functions as dnn
import tensorflow as tf
import time


#%%
"""transforming trainable variables to one array"""


def W_to_x(trainable_vars):
    sh = tf.shape(trainable_vars[0])[0] * tf.shape(trainable_vars[0])[1]
    x = tf.reshape(trainable_vars[0], [sh, 1])
    for v in trainable_vars[1:]:
        sh = tf.shape(v)[0] * tf.shape(v)[1]
        v = tf.reshape(v, [sh, 1])
        x = tf.concat([x, v], 0)
    return x


def x_to_W(x, shapes):
    trains = []
    start = 0
    for sh in shapes:
        dd = sh[0] * sh[1] + start
        xi = tf.reshape(x[start:dd], sh)
        trains.append(xi)
        start = dd
    return trains


#%%
cluster = tf.compat.v1.train.ClusterSpec({"worker": ["n03:2222", "n04:2223"]})
server00 = tf.distribute.Server(cluster, job_name="worker", task_index=1)
server0 = tf.distribute.Server(cluster, job_name="worker", task_index=0)
# server1 = tf.distribute.Server(cluster, job_name='worker', task_index=2)
# server2 = tf.distribute.Server(cluster, job_name='worker', task_index=3)


#%%
@tf.function
def calculate_hess(trainable_variables, X, d, a):
    with tf.device("/job:worker/task:0"):
        part_of_trains = trainable_variables[6:7]
        # here I test different parts, the first is the most demanding
        indices = [ind for ind in range(6, 7)]
        A0 = dnn.hes_part(x, t, indices_zero_one, trainable_variables, part_of_trains, indices)
    return A0


@tf.function
def test(trainable_variables, X, d, a):
    """Test whether the distributing works"""
    with tf.device("/job:worker/task:1"):
        dr = trainable_variables[1] + 2.0
        X = X + a * d
        vars = x_to_W(X, shapes)
        i = 0
        for tv in vars:
            trainable_variables[i].assign(tv)
            i = i + 1
        d.assign(d + 0.001)
        a.assign(a + 0.001)
    return dr


#%%
"""main loop"""

n = 272

with tf.compat.v1.Session(server00.target) as sess:
    trainable_variables = dnn.create_weights(10, 10, 0, 0.1)
    X = W_to_x(trainable_variables)
    x, t, indices_zero_one = dnn.create_input(1000, 0.2)
    shapes = []
    for v in trainable_variables:
        shapes.append(tf.shape(v))
    d0 = tf.zeros((n, 1))
    d = tf.Variable(d0)
    a = tf.Variable(1.0)
    init_op = tf.compat.v1.global_variables_initializer()
    sess.run(init_op)

    for k in range(3):
        start = time.time()
        pr = calculate_hess(trainable_variables, X, d, a)
        pr = pr[0].eval()
        dr = test(trainable_variables, X, d, a)
        dr = dr.eval()
        end = time.time()
        print("total time:", end - start)
        print("iteration", k)
        print("pr:", pr)
