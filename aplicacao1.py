import wry
import get_data
from santas_little_helpers import *
import matplotlib.pyplot as plt
import seaborn as sb
import random
import copy

random.seed(420)

data = get_data.get_data()

defns = wry.ProblemDefns()
defns.NUM_EPOCHS = 1000
defns.BATCH_SIZE = 20
defns.DATA = data

print("""#---------------------------------------------------\n# executando com as seguintes definições do
problema\n""" , defns)

results = wry.main(defns)
with open("run_example.log", 'wt') as log:
    log.write(str([i for i in results]))
print("succefully ran")
#
# fig = plt.figure()
# plt.plot(losses, 'o')
# fig.show()
#01)
