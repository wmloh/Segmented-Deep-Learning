from operation import Operation
from model import Model
from examples import AND_dataset, OR_dataset, NOT_dataset, random_3inputs
import numpy as np

# Note that 2 Operation objects may not exist at the same time due to Python
#   automatically sending the already-called object for garbage collection

'''
a, b = random_3inputs(size=500)
p = Model(3, 2, 3, activation='sigmoid')
p.train(a, b, alpha = 0.0001, showRunTime=True)

x, y = NOT_dataset()
n = Model(1, 2, 1, activation='sigmoid')
n.train(x, y, epoch=5000, alpha=0.0001, display=False)

x, y = AND_dataset()
m = Model(2, 2, 2, activation='tanh')
m.train(x, y, epoch=5000,display=False)

o = Operation(2, 2)
o.add(m)
o.add(n)

X, y = AND_dataset()
m = Model(2,2,1)
m.train(X, y)
k, l = AND_dataset()
n = m.test(k, l)
print(n)
'''
