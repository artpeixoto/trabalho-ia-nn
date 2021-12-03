import tensorflow as tf2
import tensorflow.keras as keras
from tensorflow.keras import (losses, layers, metrics, optimizers)
import numpy as np
import pandas
from pathlib import Path
import os

rede = keras.Sequential([
        layers.InputLayer(3),
        layers.Dense(units= 10, activation="sigmoid"),
        layers.Dense(units=1, activation="sigmoid")
        ]
    )


rede.compile(
        optimizer=
            optimizers.SGD(learning_rate=0.1),
        loss=keras.losses.MeanSquaredError()
    )



#01)
#treinando
for i in range(5)