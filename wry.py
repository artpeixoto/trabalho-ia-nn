import numpy as np
import tensorflow as tf
import random
import copy
from functools import *
from itertools import *
from more_itertools import *
import dataclasses

from get_data import *
from arturs_little_helpers import *


from tensorflow.python.ops.numpy_ops import np_config
np_config.enable_numpy_behavior()



# #---------------------------------------------------
# # definições do problema

# DTYPE = tf.float64 #para garantir a precisão requerida, usamos o maior float disponivel
# DATA_SIZES = (3,10,1)
# ACTIVATION_FUNCTION = lambda parms, x: tf.sigmoid(parms @ x )
# BATCH_SIZE = 200
# NUM_EPOCHS = 5
# RANDOM_PARMS_GENERATOR = lambda shape: tf.random.uniform(shape=shape, dtype=DTYPE, minval=-0.5, maxval=+0.5)
# LOSS_FUNCTION = lambda errs: tf.math.reduce_mean(tf.square(errs))
# UPDATE_RATIO = 0.1
# #---------------------------------------------------


@dataclasses.dataclass
class ProblemDefns:
    NEURAL_NET = None
    NEURAL_NET_ACTIVATIONS = repeat(lambda parms, x: tf.sigmoid(parms @ x))
    DATA_SIZES = (3, 10, 1)
    RANDOM_PARMS_GENERATOR = lambda shape, **kwargs: tf.random.uniform(shape=shape, minval=-0.5, maxval=+0.5, **kwargs)
    PARMS = None
    DTYPE = tf.float64
    BATCH_SIZE = 200
    NUM_EPOCHS = 5
    LOSS_FUNCTION = lambda errs: tf.math.reduce_mean(tf.square(errs))
    UPDATE_RATIO = 0.1

def make_neural_net_function(activation_fns,): # star_args_list=repeat(None), start_kwargs_list):
    layers = activation_fns
    def neural_net(parms, x):
        activations = [x]
        for parm, layer in zip(parms, layers):
            activation=activations[-1]
            activations.append(layer(parms, activation))
        return activation, activations
    return neural_net

def generate_parms_shapes (data_sizes):
    return (sh[::-1] for sh in  pairwise(iter(data_sizes)))

def generate_parms_generator(generator_fn):
    return lambda shapes, **kwargs: [generator_fn(shape=sh, **kwargs) for sh in shapes]

def shuffle_dataset(dataset):
    rand_indexes = random.sample(range(len(dataset)), len(dataset))
    return dataset[rand_indexes]
    
def get_batches_generator(n, ds):
    li = 0
    while True:
        ui = min(len(ds), li + n)
        if ui == li: break
        else:
            yield ds[li:ui]
            li = ui

def sgd(parm, grad, update_amount, *args, **kwargs): #descida de  gradiente
    return parm - update_amount*grad


def train_nn(neural_net, parms, x_train, y_train, loss_function, parm_train_fn):
    with tf.GradientTape() as tape:
        tape.watch(parms)
        y_pred = neural_net(parms, x_train)
        errors = y_train - y_pred
        losses = loss_function(errors)
    parm_grads = tape.gradient(losses, parms)
    new_parms = parm_train_fn(parms, parm_grads, losses)
    return locals()

def main(defns: ProblemDefns):

    parms = defns.PARMS
    if not parms:
        shapes = generate_parms_shapes(defns.DATA_SIZES)
        parms = (generate_parms_generator(defns.RANDOM_PARMS_GENERATOR))(shapes, dtype=defns.DTYPE)
    neural_net = defns.NEURAL_NET
    if not neural_net:
        neural_net = tf.function(make_neural_net_function(defns.NEURAL_NET_ACTIVATIONS))
    epochs_results= []
    for epoch in range(defns.NUM_EPOCHS):
        batches_results = []
        for batch in get_batches_generator(defns.BATCH_SIZE, shuffle_dataset(train))):
            tensor_batch = tf.constant(batch)
            train_nn(
                neural_net = neural_net, \
                parms = parms, \
                x_train, \
                y_train, \
                loss_function, \
                parm_train_fn
            )
        epochs_results.append(batches_results)
    return locals()

if __name__ == "__main__":
    locals().update(main())