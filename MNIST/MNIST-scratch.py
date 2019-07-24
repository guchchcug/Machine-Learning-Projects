# This is a mnist solution using a standard fully connected network
# with two hidden layers.

from __future__ import print_function

import numpy as np
import random
import keras
from keras.datasets import mnist

N_classes = 10
dim = 784
N_train = 60000
N_test = 10000

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.reshape(N_train, dim)
x_test = x_test.reshape(N_test, dim)
x_train = x_train.astype('float32')  # change to float32, because we'll divide that by 255 later.
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

y_train = keras.utils.to_categorical(y_train, N_classes)
y_test = keras.utils.to_categorical(y_test, N_classes)

# Append the implicit bias term to the data so we don't have to worry
# about having an extra parameter vector
x_train = np.concatenate((x_train, np.ones((N_train, 1))), axis=1)
x_test = np.concatenate((x_test, np.ones((N_test, 1))), axis=1)

# Initialize the model - use implicit biases (the +1)
N_hidden_1 = 64
N_hidden_2 = 64
Wh1 = np.random.rand(N_hidden_1, dim + 1) / (dim + 1)  # (65, 785)
Wh2 = np.random.rand(N_hidden_2, N_hidden_1 + 1) / (N_hidden_1 + 1)  # (65, 65)
Wo = np.random.rand(N_classes, N_hidden_2 + 1) / (N_hidden_2 + 1)  # (10, 65)

# Here's the training loop
iteration = 100000
step_size = 0.001
momentum = 0.9
train_loss = np.zeros(iteration)
v_dL_dWo = np.zeros(Wo.shape)
v_dL_dWh1 = np.zeros(Wh1.shape)
v_dL_dWh2 = np.zeros(Wh2.shape)

for i in range(iteration):
    idx = random.randint(0, N_train-1)
    x = np.reshape(x_train[idx], [-1, 1])  # (785, 1)
    y = np.reshape(y_train[idx], [-1, 1])  # (785, 1)
    # forward pass the batch to the first hidden layer
    hb1 = np.dot(Wh1, x)  # (64, 1)
    hb1 = np.concatenate((hb1, np.ones([1, 1])))  # (65, 1)
    h1 = np.maximum(np.zeros([N_hidden_1 + 1, 1]), hb1)  # (65, 1)
    # Forward pass the batch to the second hidden layer
    hb2 = np.dot(Wh2, h1)  # (64, 1)
    hb2 = np.concatenate((hb2, np.ones([1, 1])))  # (65, 1)
    h2 = np.maximum(np.zeros([N_hidden_2 + 1, 1]), hb2)  # (65, 1)
    o = np.dot(Wo, h2)  # (10, 1)
    o_shift = o - np.amax(o)
    o_shift_exp = np.exp(o_shift)
    o_shift_exp_sum = sum(o_shift_exp)
    p = o_shift_exp / o_shift_exp_sum
    L = - sum(y * np.log(p))
    train_loss[i] = L
    if i % 1000 == 0:
        print(L)
    # Now backprop gradients to find dL_dWo and dL_dWh
    dL_dp = - y / p
    dp_do = (np.diag(o_shift_exp.flatten() * o_shift_exp_sum) - np.tile(o_shift_exp, N_classes) *
             np.tile(o_shift_exp.T, [N_classes, 1])) / o_shift_exp_sum ** 2
    dL_do = np.dot(dL_dp.T, dp_do).T
    dL_dWo = dL_do * np.tile(h2.T, [N_classes, 1])
    dL_dh2 = np.dot(dL_do.T, Wo).T
    dL_dhb2 = dL_dh2 * np.reshape((h2 == hb2), [-1, 1])
    dL_dWh2 = dL_dhb2[:-1] * np.tile(h1.T, [N_hidden_1, 1])
    dL_dh1 = np.dot(dL_dhb2[:-1].T, Wh2).T
    dL_dhb1 = dL_dh1 * (h1 == hb1)
    dL_dWh1 = dL_dhb1[:-1] * np.tile(x.T, [N_hidden_2, 1])
    # Take a gradient step to update Wo and Wh
    v_dL_dWo = step_size * dL_dWo + momentum * v_dL_dWo
    v_dL_dWh1 = step_size * dL_dWh1 + momentum * v_dL_dWh1
    v_dL_dWh2 = step_size * dL_dWh2 + momentum * v_dL_dWh2
    Wo = Wo - v_dL_dWo
    Wh1 = Wh1 - v_dL_dWh1
    Wh2 = Wh2 - v_dL_dWh2

# Calculate accuracy
# Now compute the testing accuracy
num_correct_test = 0
for idx in range(N_test):
    x = np.reshape(x_test[idx], [-1, 1])  # (785, 1)
    y = np.reshape(y_test[idx], [-1, 1])
    # forward pass the batch to the first hidden layer
    hb1 = np.dot(Wh1, x)  # (64, 1)
    hb1 = np.concatenate((hb1, np.ones([1, 1])))  # (65, 1)
    h1 = np.maximum(np.zeros([N_hidden_1 + 1, 1]), hb1)  # (65, 1)
    # Forward pass the batch to the second hidden layer
    hb2 = np.dot(Wh2, h1)  # (64, 1)
    hb2 = np.concatenate((hb2, np.ones([1, 1])))  # (65, 1)
    h2 = np.maximum(np.zeros([N_hidden_2 + 1, 1]), hb2)  # (65, 1)
    o = np.dot(Wo, h2)  # (10, 1)
    o_shift = o - np.amax(o)
    # Notice that we use a trick here to avoid overflow when taking the
    # exp() of o. We shift all of the entries of o by max(o), then
    # exponentiate. The resulting p is mathematically equivalent to the
    # p we would obtain if we just exponentiated the raw o values, but
    # on a real computer with limited range it's safer to do it this way
    o_shift_exp = np.exp(o_shift)
    o_shift_exp_sum = sum(o_shift_exp, 1)
    p = o_shift_exp / o_shift_exp_sum
    prediction = np.argmax(p)
    truth = np.argmax(y)
    if prediction == truth:
        num_correct_test = num_correct_test + 1

test_accuracy = num_correct_test / N_test
print(test_accuracy)
