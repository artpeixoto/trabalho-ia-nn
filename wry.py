import numpy as np
import tensorflow as tf
from tensorflow.python.ops.numpy_ops import np_config
np_config.enable_numpy_behavior()
import logging 

from functools import *
from itertools import *
from sys import *


#"ah que fazer importacao total eh feio" entao me chama de feio mas eu importo as biblioteca padrao mesmo sem do. Se acha ruim vai ali, ta vendo? bem ali pra casa do caralho


debug = True

#é importante notar que, nesse arquivo, eu não tenho intenção de me ater as convenções de codigo pythonico, mas estou programando com paradima funcional. Isso pois estou estudando linguagens funcionais (haskell) e alem de apresentarem uma alternativa mais segura, eficiente de codigo, preciso de treino nelas.


#===================================================


#---------------------------------------------------
neural_layer = lambda fn, parms, x: (
    fn( parms.T @ x) #note que com uma definição funcional como essa, qualquer implementacao das operacoes basicas pode ser usada, desde que se atenham a convencao da linguagem (foo.__matmul__)
)

def neural_net_function(activation_fns, parms, input):
    first_layer, second_layer = (neural_layer(activation_fn)(parm) for parm, activation_fn in zip(parms, activation_fns))
    return second_layer(first_layer(input))

def error_fn(error_fn, y_true, y_pred):
    return error_fn(y_true-y_pred)
#---------------------------------------------------





#---------------------------------------------------
def tee(iterable):
    from copy import deepcopy
    return iterable, deepcopy(iterable)

def pairwise(iterable): #funcao ajudante roubada- digo, emprestada de https://docs.python.org/3/library/itertools.html#itertools.pairwise
    # pairwise('ABCDEFG') --> AB BC CD DE EF FG
    a, b = tee(iterable)
    next(b, None)
    return zip(a, b)

def take_n(n, xs):
    if len(xs) <= n:
        yield xs
    else:
        yield xs[:n]
        yield from take_n(n, xs[n:])

get_parms_shapes = lambda data_sizes: pairwise(iter(data_sizes))

generate_parms = lambda build_fn: lambda shapes: [build_fn(shape=sh) for sh in shapes]
#---------------------------------------------------





#---------------------------------------------------
# um pouco de debug

#criamos um pequeno ajudante para xeretar alguma funcao qualquer.

def verboiser(message_function, original_function):
        def wrapper(*args, **kwargs):
            res =  original_function(*args, **kwargs)
            message_function(res, *args, **kwargs,)
            return res
        return wrapper


logger = logging.getLogger()
logger.addHandler(logging.Handler(stdout))
log_obj = lambda obj: logger.info(str(obj))
#---------------------------------------------------


#---------------------------------------------------
# funcoes para lidar com os dados
import random
random.seed(137)

shuffle_dataset = lambda ds: (i for i in random.shuffle(ds))
get_batches = lambda n: lambda ds: [i for i in take_n(n, ds)]
#---------------------------------------------------


#====================================================

#temos todas as pecas basicas, e podemos comecar a montar nossas experiencias

#---------------------------------------------------
# definições do problema

DTYPE = tf.float64 #para garantir a precisão requerida, usamos o maior float disponivel
DATA_SIZES = (3,10,1)
ACTIVATION_FUNCTION = tf.sigmoid
BATCH_SIZE = 1
RANDOM_PARMS_GENERATOR = lambda shape: tf.random.uniform(shape=shape, dtype=DTYPE, seed=137, minval=-10, maxval=+10)
ERROR_FN = lambda err: tf.mean(tf.square(err))

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

[(x_train, y_train), (x_test, y_test)] = [
    (lambda t: (t[:,1:4], t[:,4][:,np.newaxis])) (table.to_numpy())
    for table in data]
#---------------------------------------------------


#---------------------------------------------------
shapes = get_parms_shapes(DATA_SIZES)
initial_parms = generate_parms(RANDOM_PARMS_GENERATOR)(shapes)

log_obj(initial_parms)
#---------------------------------------------------


#---------------------------------------------------
logger.info("Building Neural Net...")
initial_neural_net = lambda x: neural_net_function(repeat(ACTIVATION_FUNCTION), initial_parms, x.T)

logger.info("Running it...")
try: log_obj(initial_neural_net(x_train))
except: 
    logger.error("oh man, didn't work")
    raise
#---------------------------------------------------


quit()