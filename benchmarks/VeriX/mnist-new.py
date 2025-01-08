import sys
sys.path.append("/home/marabou2/artifact/tools/Marabou")

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
parser.add_argument('--network', type=str, default='/home/marabou2/artifact/runtime_evaluation/benchmarks/VeriX/mnist2x10')
parser.add_argument('--index', type=int, default=0)
parser.add_argument('--epsilon', type=float, default=0.05)
parser.add_argument('--traverse', type=str, default='gradient')
parser.add_argument('--seed', type=int, default=0)
args = parser.parse_args()

dataset = args.dataset
model_name = args.network
index = args.index
epsilon = args.epsilon
traverse = args.traverse
seed = args.seed

if traverse == 'heuristic' or 'gradient':
    result_dir = 'index-%d-%s-%ds-%s-linf%g' % (
        index, model_name, TIMEOUT, traverse, epsilon)
elif traverse == 'random':
    result_dir = 'index-%d-%s-%ds-%s-seed-%d-linf%g' % (
        index, model_name, TIMEOUT, traverse, seed, epsilon)
else:
    print('traversal incorrect.')
    exit()

(x_train, y_train), (x_test, y_test) = mnist.load_data(path="/home/marabou2/artifact/runtime_evaluation/benchmarks/VeriX/mnist.npz")
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
print("Test loss:", score[0])
print("Test accuracy:", score[1])

logits = keras_model.predict(np.expand_dims(x[index], axis=0))
label = logits.argmax()
print(logits)
print(label)

if label != y[index].argmax():
    print("Wrong prediction. Pass")
    exit()

explanation_tick = time.time()

onnx_model_path = model_name + '.onnx'
mara_network = Marabou.read_onnx(onnx_model_path)
# options = Marabou.createOptions(numWorkers=16, timeoutInSeconds=TIMEOUT, verbosity=0)
options = Marabou.createOptions(timeoutInSeconds=TIMEOUT,
                                verbosity=0)

inputVars = mara_network.inputVars[0][0].flatten()
outputVars = mara_network.outputVars[0].flatten()

if traverse == 'heuristic':
    # heuristic: get traverse order by pixel sensitivity
    temp = x[index].reshape(28*28)
    image_batch = np.kron(np.ones((28*28, 1)), temp)
    image_batch_manipulated = image_batch.copy()
    for i in range(28*28):
        image_batch_manipulated[i][i] = 1 - image_batch_manipulated[i][i]
        # image_batch_manipulated[i][i] = 0
    predictions = keras_model.predict(image_batch.reshape(784, 28, 28, 1))
    predictions_manipulated = keras_model.predict(image_batch_manipulated.reshape(784, 28, 28, 1))
    difference = predictions - predictions_manipulated
    features = difference[:, label]
    sorted_index = features.argsort()
    inputVars = sorted_index

    sensitivity = features.reshape([28, 28])
    # np.savetxt('%s/index-%d-%s-linf%g-saliency.txt' % (result_dir, index, model_name, epsilon),
               # np.flip(inputVars), fmt='%d')
    # exit()

elif traverse == 'gradient':
    temp = tf.cast(np.expand_dims(x[index], axis=0), tf.float32)
    with tf.GradientTape() as tape:
        tape.watch(temp)
        prediction = keras_model(temp)
        top_class = prediction[:, logits.argmax()]
    gradients = tape.gradient(top_class, temp)
    gradient = gradients[0].numpy()
    sorted_index = gradient.flatten().argsort()
    sorted_index = sorted_index[::-1]
    inputVars = sorted_index

elif traverse == 'random':
    import random
    random.seed(seed)
    random.shuffle(inputVars)


print(inputVars)

image = x[index].flatten()

unsat_set = []
sat_set = []
timeout_set = []

marabou_time = []

for pixel in inputVars:
    for j in range(10):
        if j != label:
            network = Marabou.read_onnx(onnx_model_path)
            network.addInequality([outputVars[label], outputVars[j]],
                                  [1, -1], -1e-6)
            # network.addInequality([outputVars[label], outputVars[j]],
            #                       [1, -1], 0)
            for i in inputVars:
                if i == pixel or i in unsat_set:
                    # network.setLowerBound(i, 0)
                    # network.setUpperBound(i, 1)
                    network.setLowerBound(i, max(0, image[i] - epsilon))
                    network.setUpperBound(i, min(1, image[i] + epsilon))
                else:
                    network.setLowerBound(i, image[i])
                    network.setUpperBound(i, image[i])
            marabou_tick = time.time()
            exitCode, vals, stats = network.solve(options=options, verbose=False)
            marabou_toc = time.time()
            marabou_time.append(marabou_toc - marabou_tick)
            if exitCode == 'sat' or exitCode == 'TIMEOUT':
                break
            elif exitCode == 'unsat':
                continue

    if exitCode == 'unsat':
        print('location %d returns unsat, move out.' % pixel)
        unsat_set.append(pixel)
        print('current outside', unsat_set)
    elif exitCode == 'TIMEOUT':
        print('timeout for pixel', pixel)
        print('do not move out, continue to the next pixel')
        timeout_set.append(pixel)
    elif exitCode == 'sat':
        print('perturbing current outside + this location %d alters prediction' % pixel)
        print('do not move out, continue to the next pixel')
        sat_set.append(pixel)

        # # adversary = [vals.get(i) for i in inputVars] ???????
        # adversary = [vals.get(i) for i in mara_network.inputVars[0][0].flatten()]
        # adversary = np.asarray(adversary).reshape(28, 28)
        # prediction = [vals.get(i) for i in outputVars]
        # prediction = np.asarray(prediction).argmax()
        #
        # plot_figure(image=adversary,
        #             path='%s/index-%d-adversary-sat-pixel-%d-predicted-as-%d.png' %
        #                  (result_dir, index, pixel, prediction),
        #             cmap='gray')

    explanation_toc = time.time()
    if pixel == inputVars[-1]:
        mask = np.zeros(image.shape).astype(bool)
        # mask[unsat_set] = 1
        mask[sat_set] = True
        mask[timeout_set] = True
        # mask = mask.astype('int')
        mask = np.zeros(image.shape).astype(bool)
        # mask[unsat_set] = 1
        mask[sat_set] = True
        mask[timeout_set] = True
        # mask = mask.astype('int')
        print("Unsat set:", unsat_set)
        print("Sat set:", sat_set)
        print("Timeout set:", timeout_set)
print("Complete!")
