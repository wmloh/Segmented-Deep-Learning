from operation import Operation
from model_mult import Model
from datasets import *
import numpy as np

m = Model(2,2,[1])
X, y = OR_dataset(size=200)
m.train(X, y)




