import keras
import numpy as np
from datatypes import *


def get_dataset(path, parsing_fn):
    with open(path, 'r') as f:
        ds = parsing_fn(f)
    return np.array()