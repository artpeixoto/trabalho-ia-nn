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
defns.NUM_EPOCHS = 100
defns.BATCH_SIZE = 20
train_data, test_data = defns.DATA = data

print("""#---------------------------------------------------\n# executando com as seguintes definições do
problema\n""" , defns)


results, neural_net = wry.main(defns)
results_df = pd.DataFrame(results)


with open("run_example.log", 'wt') as log:
    log.write(str([i for i in results]))
print("succefully ran training.")

fig = plt.figure()
plt.plot(results_df.losses, 'o')
fig.show()
#01)
