import numpy as np
import tensorflow as tf
import random
from functools import *
from itertools import *
from more_itertools import *
import attr
from numpy import array
from santas_little_helpers import *


from tensorflow.python.ops.numpy_ops import np_config


@attr.s
class ProblemDefns:
    NEURAL_NET = attr.ib(default=None)
    NEURAL_NET_ACTIVATIONS = attr.ib(default=repeat(lambda parms, x: tf.sigmoid(tf.matmul(parms, x))))
    DATA_SIZES = attr.ib(default=(3, 10, 1))
    RANDOM_PARMS_GENERATOR = attr.ib(default=lambda **kwargs: tf.random.uniform(minval=-0.5, maxval=+0.5, **kwargs))
    PARMS = attr.ib(default=None)
    DTYPE = attr.ib(default=tf.float64)
    BATCH_SIZE = attr.ib(default=20)
    NUM_EPOCHS = attr.ib(default=5)
    LOSS_FUNCTION = attr.ib(default=lambda errs: tf.math.reduce_mean(tf.square(errs)))
    UPDATE_RATIO = attr.ib(default=0.1)
    DATA = attr.ib(default=None)
    

def make_neural_net_function(activation_fns,): # star_args_list=repeat(None), start_kwargs_list):
    layers = activation_fns
    def neural_net(parms, x):
        activation = x
        for parm, layer in zip(parms, layers):
            activation = layer(parm, activation)
        return activation
    return neural_net



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


np_config.enable_numpy_behavior()

def generate_parms_shapes (data_sizes):
    return [sh[::-1] for sh in pairwise(iter(data_sizes))]
    
def generate_parms_generator(generator_fn):
    return lambda shapes, **kwargs: [generator_fn(shape=sh, **kwargs) for sh in shapes]

def get_random_indexes(dataset_len):
    rand_indexes = random.sample(range(dataset_len),dataset_len)
    return rand_indexes


def get_batches_generator(n, ds):
    li = 0
    while True:
        ui = min(len(ds), li + n)
        if ui == li:
            break
        else:
            yield ds[li:ui]
            li = ui

def sgd(parm, grad, update_amount, *args, **kwargs): #descida de  gradiente
    return parm - update_amount*grad

def nn_evaluate_execution(neural_net, parms, x_train, y_train, loss_function):
    with tf.GradientTape() as tape:
        tape.watch(parms)
        y_pred = neural_net(parms, x_train)
        errors = y_train - y_pred
        losses = loss_function(errors)
    parm_grads = tape.gradient(losses, parms)
    return dict(
        x_train = x_train,
        y_train = y_train,
        y_pred = y_pred,
        errors = errors,
        losses = losses,
        parm_grads = parm_grads,
    )

def main(defns: ProblemDefns):
    (x_train, y_train), (x_test, y_test)  = train, test = defns.DATA
    parms = defns.PARMS
    if not parms:
        shapes = generate_parms_shapes(defns.DATA_SIZES)
        parms = (generate_parms_generator(defns.RANDOM_PARMS_GENERATOR)(shapes=shapes, dtype=defns.DTYPE))
    neural_net = defns.NEURAL_NET
    if not neural_net:
        neural_net = tf.function(make_neural_net_function(defns.NEURAL_NET_ACTIVATIONS))
    epochs_results= []
    #main looping
    loop_n = 0
    for epoch in range(defns.NUM_EPOCHS):
        batches_results = []
        shuffle_indexes = get_random_indexes(len(x_train.T))
        for x_train_batch, y_train_batch in zip(get_batches_generator(defns.BATCH_SIZE, x_train.T[shuffle_indexes]), \
                get_batches_generator(defns.BATCH_SIZE, y_train[shuffle_indexes])):
            
            execution_results = nn_evaluate_execution(
                neural_net = neural_net, \
                parms = parms, \
                x_train = x_train_batch.T, \
                y_train = y_train_batch, \
                loss_function = defns.LOSS_FUNCTION, \
                )
            i_parms = parms
            iplus1_parms = [sgd(parm, grad, update_amount=defns.UPDATE_RATIO)
                            for parm, grad in zip(parms, execution_results["parm_grads"])]
            parms = iplus1_parms
            batches_results.append(dict(
                loop_n=loop_n,
                parms = i_parms,
                next_parms = iplus1_parms,
                **execution_results
                )
            )
            loop_n += 1
        epochs_results += batches_results
    return epochs_results, neural_net

if __name__ == "__main__":
    basic_definitions = ProblemDefns()
    import get_data
    basic_definitions.DATA = [[array(j) for j in i] for i in get_data.get_data()]
    print(main(basic_definitions))
    