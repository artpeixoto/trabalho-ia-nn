import keras
from keras import *
import numpy as np
from datatypes import *


def get_dataset(path, parser):
    with open(path, 'r') as f:
        raw_data = f.read()
    x_train, y_train, x_test, y_test = parser(raw_data)
    return x_train, y_train, x_test, y_test

