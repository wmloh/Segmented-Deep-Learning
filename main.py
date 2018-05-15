from operation import Operation
from model_mult import Model
from examples import AND_dataset, OR_dataset, NOT_dataset, random_3inputs
import numpy as np

# Note that 2 Operation objects may not exist at the same time due to Python
#   automatically sending the already-called object for garbage collection

m = Model(2,2,[5,4,3])
X, y = AND_dataset()
m.train(X, y)
print(m.predict([1,1]))
print(m.predict([1,0]))
print(m.predict([0,1]))
print(m.predict([0,0]))
