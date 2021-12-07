import numpy as np
import tensorflow as tf
from tensorflow.python.ops.numpy_ops import np_config
np_config.enable_numpy_behavior()
import logging 
import copy

from functools import *
from itertools import *
from sys import *
from more_itertools import *

logging_level = logging.DEBUG

#---------------------------------------------------
# um pouco de debug

#criamos um pequeno ajudante para xeretar alguma funcao qualquer.

def verboiser(message_function, original_function):
        def wrapper(*args, **kwargs):
            res =  original_function(*args, **kwargs)
            message_function(res, *args, **kwargs,)
            return res
        return wrapper


rlogger = logging.RootLogger(logging_level)
rlogger.addHandler(logging.StreamHandler(stdout))

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

#é importante notar que, nesse arquivo, eu não tenho intenção de me ater as convenções de codigo pythonico, mas estou programando com paradima funcional. Isso pois estou estudando linguagens funcionais (haskell) e alem de apresentarem uma alternativa mais segura, eficiente de codigo, preciso de treino nelas.

def neural_net_function(activation_fns, parms, x):
    layers = (lambda x: activation_fn(parm, x) for parm, activation_fn in zip(parms, activation_fns))
    activation = x
    for l in layers:
        activation = l(activation)
    return activation

get_parms_shapes = lambda data_sizes: (sh[::-1] for sh in  pairwise(iter(data_sizes)))
generate_parms = lambda build_fn: lambda shapes: [build_fn(shape=sh) for sh in shapes]
#---------------------------------------------------


#---------------------------------------------------
# funcoes para lidar com os dados
import random
random.seed(137)

#---------------------------------------------------


#---------------------------------------------------
# definições do problema

DTYPE = tf.float64 #para garantir a precisão requerida, usamos o maior float disponivel
DATA_SIZES = (3,10,1)
ACTIVATION_FUNCTION = tf.function(lambda parms, x: tf.sigmoid(parms @ x))
BATCH_SIZE = 64
NUM_EPOCHS = 5
RANDOM_PARMS_GENERATOR = lambda shape: tf.random.uniform(shape=shape, dtype=DTYPE, seed=137, minval=-10, maxval=+10)
LOSS_FUNCTION = tf.function(lambda errs: np.mean(tf.square(errs)))

#---------------------------------------------------


#---------------------------------------------------
# importando os dados
from pathlib import Path
import pandas
here = Path(".")

dataset_path = here / "sources" / "RNA - Projeto - Dados" / "DadosProjeto01RNA.xlsx"
dataset_file = open(dataset_path, "rb")
data = [
        pandas.read_excel(
            io=dataset_file,
            sheet_name=sheet
            )
        for sheet in ["DadosTreinamentoRNA", "DadosTesteRNA"]]
logger.debug(data)
train, test = [table.to_numpy() for table in data]
logger.debug(test, train)


shapes = get_parms_shapes(DATA_SIZES)
initial_parms = generate_parms(RANDOM_PARMS_GENERATOR)(shapes)

print((initial_parms))

logger.info("Building Neural Net...")
initial_neural_net = tf.function(lambda x: neural_net_function(repeat(ACTIVATION_FUNCTION), initial_parms, x))

separate_xy = tf.function((lambda t: (t[:,1:4].T, t[:,4])))


def shuffle_dataset(dataset):
    rand_indexes = random.sample(range(len(dataset)), len(dataset))
    return dataset[rand_indexes]
     

def get_batches(n, ds):
    li = 0
    while True:
        ui = min(len(ds), li + n)
        if ui == li: break
        else:
            yield ds[li:ui]
            li = ui


@tf.function
def evaluate_nn(neural_net, loss_function, batch):
    x_train, y_train = separate_xy(batch)
    y_pred = neural_net(x_train)
    errors = y_train - y_pred
    losses = loss_function(errors)
    return y_pred, errors, losses

neural_net = initial_neural_net
epoch_loop = []


for epoch in range(NUM_EPOCHS):
    train_ds = shuffle_dataset(train)
    batch_loop = []
    for batch in get_batches(BATCH_SIZE, train_ds):
        print(batch)
        res = evaluate_nn(neural_net, LOSS_FUNCTION, batch)
        print(res)
        batch_loop.append(res)
    epoch_loop.append(batch_loop)


logger.info("Running it...")
try: log_obj(initial_neural_net(x_train))
except: 
    logger.error("oh man, didn't work")
    raise
