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

#importando os dados
here = Path()

dataset_path = here / "sources" / "RNA - Projeto - Dados" / "DadosProjeto01RNA.xlsx"
dataset_file = open(dataset_path, "rb")
data = [
        pandas.read_excel(
            io=dataset_file,
            sheet_name=za_sheeeit
            )
        for za_sheeeit in ["DadosTreinamentoRNA", "DadosTesteRNA"]
    ]

print(data[0].head())

outer = []
for tabela in map(lambda x: x.to_numpy(), data):
    x, y = [tabela[:, 1:4], tabela[5]]
    outer.append([x,y])