import wry
import get_data
import tensorflow as tf
from santas_little_helpers import *
import matplotlib.pyplot as plt
import seaborn as sb
import random
import copy
import pandas as pd
from santas_little_helpers import *

random.seed(420)

data = get_data.get_data()

defns = wry.ProblemDefns()
defns.NUM_EPOCHS = 20
defns.BATCH_SIZE = 100
train_data, test_data = defns.DATA = data

with open("results/run_example.log", 'wt') as log:
    log.write("""#---------------------------------------------------\n# executando com as seguintes definições do
        problema\n""" + str(defns))
    results, neural_net = wry.main(defns)
    fig = plt.figure()
    for treinamento in range(5):
        log.write("""#---------------------------------------------------\n# executando com as seguintes definições do
        problema\n""" + str(defns))
        results_df = pd.DataFrame(results)
        latest_parms = results_df.next_parms[len(results_df.next_parms)-1]

        
        log.write(str([i for i in results_df.iterrows()]))
        print(f"succefully ran training #{treinamento}.")
        plt.plot(results_df.losses.apply(np.array), 'o', alpha=.6)
        defns.PARMS = latest_parms
        defns.NEURAL_NET = neural_net
        results, neural_net = wry.main(defns)

#01)
