_from pathlib import Path
import pandas
import tensorflow as tf
#---------------------------------------------------
# importando os dados

from tensorflow.python.ops.numpy_ops import np_config
np_config.enable_numpy_behavior()

here = Path(".")

dataset_path = here / "sources" / "RNA - Projeto - Dados" / "DadosProjeto01RNA.xlsx"
dataset_file = open(dataset_path, "rb")
data = [
        pandas.read_excel(
            io=dataset_file,
            sheet_name=sheet
            )
        for sheet in ["DadosTreinamentoRNA", "DadosTesteRNA"]]

#-------------------------------------------------
#funcao ajudante
def separate_xy (t):
    return  (t[:,1:4].T, t[:,4])
#-------------------------------------------------
def get_data():
    train, test = [separate_xy( tf.constant(table.to_numpy())) for table in data]
    return train, test