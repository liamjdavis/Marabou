import sys
sys.path.append("/barrett/scratch/haozewu/marabou2/Marabou/")

import os
import time
import random
import argparse
import numpy as np
import tensorflow as tf
import tensorflow_ranking as tfr
from keras.datasets import mnist
from keras.models import load_model
from skimage.color import label2rgb
import matplotlib.pyplot as plt
from maraboupy import Marabou


TIMEOUT = 300

# directory = 'rebuttal/'
# if not os.path.exists(directory):
#     os.mkdir(directory)

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='mnist')
parser.add_argument('--network', type=str, default='/barrett/scratch/haozewu/marabou2/ipqs/xai/mnist2x10')
parser.add_argument('--epsilon', type=float, default=0.05)
parser.add_argument('--traverse', type=str, default='gradient')
parser.add_argument('--seed', type=int, default=0)
args = parser.parse_args()

dataset = args.dataset
model_name = args.network
epsilon = args.epsilon
traverse = args.traverse
seed = args.seed

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
y_train = tf.keras.utils.to_categorical(y_train, 10)
y_test = tf.keras.utils.to_categorical(y_test, 10)
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255
x = x_test
y = y_test

keras_model_path = model_name + '.h5'
keras_model = load_model(keras_model_path)
keras_model.summary()
# keras_model.compile(loss=tfr.keras.losses.SoftmaxLoss(),
#                     optimizer=tf.keras.optimizers.Adam(),
#                     metrics=['accuracy'])
score = keras_model.evaluate(x, y, verbose=0)

correct = 0
with open("correctly_classified_index.txt", 'w') as out_file:
    for index in range(2000):
        logits = keras_model.predict(np.expand_dims(x[index], axis=0))
        label = logits.argmax()

        if label != y[index].argmax():
            continue
        else:
            out_file.write(f"{index}\n")
            correct += 1
        if correct == 100:
            exit(0)
