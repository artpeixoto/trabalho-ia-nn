import wry
from santas_little_helpers import *
import matplotlib.pyplot as plt
import seaborn as sb
import random
import copy

random.seed(420)
#---------------------------------------------------
# definições do problema

args =  copy.deepcopy(wry.base_args) 
args["BATCH_SIZE"] = 200
args["NUM_EPOCHS"] = 100

results = wry.main(**args)
losses = wry.get_losses(results)
fig = plt.figure()
plt.plot(losses, 'o')
fig.show()
#01)
