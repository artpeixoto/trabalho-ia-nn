import logging 
import sys
#---------------------------------------------------
# um pouco de debug
from itertools import *
from more_itertools import *
from sys import *
from functools import *
import collections
from abc import *
from collections.abc import *

import numpy as np
import tensorflow as tf

logging_level = logging.INFO

#criamos um pequeno ajudante para xeretar alguma funcao qualquer.

def recursive_untensorify(stuff):
    if isinstance(stuff, tf.Tensor):
        return stuff.numpy()
    elif isinstance(stuff, Iterable):
        return np.array([recursive_untensorify(i) for i in stuff])
    else:
        return stuff

def verboiser(message_function, original_function):
        def wrapper(*args, **kwargs):
            res =  original_function(*args, **kwargs)
            message_function(res, *args, **kwargs,)
            return res
        return wrapper


rlogger = logging.RootLogger(logging_level)
logger = logging.getLogger()
logger.setLevel(logging_level)
logger.addHandler(logging.StreamHandler(sys.stdout))
#---------------------------------------------------
