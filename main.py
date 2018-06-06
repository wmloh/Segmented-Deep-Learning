from operation import Operation
from model import Model
from datasets.logical import *
from datasets.random import *
from util.diagnosis import *
from sklearn.datasets import *
'''
O = Operation(4, 2)

m1 = Model(2,2,[2])
m2 = Model(2,2,[2])
m3 = Model(2,2,[2])

X, y = AND_dataset()
A, b = OR_dataset()

m1.train(X, y, epoch=1000)
m2.train(A, b, epoch=1000)
m3.train(X, y, epoch=1000)

O.add(m1, 1)
O.add(m2, 1)
O.add(m3, 2)

print(O([1,0,1,0]))
'''
'''
m = Model(2,2,[3], activation='sigmoid')
X, y = make_circles(n_samples=300, noise=0)
A, b = make_circles(n_samples=50, noise=0)
m.train(X, y, alpha=0.01, epoch=500, reg_strength = 0, showLossCount=100)
print(m.validate(A, b))
m.plot(X, y)
'''
'''
m = Model(2,2,[4,4], activation='sigmoid')
n = Model(2,2,[4,4], activation='tanh')
X, y = pure_random(25)
m.train(X, y, alpha=0.008, epoch=10000, reg_strength=0)
n.train(X, y, alpha=0.008, epoch=10000)
m.plot(X,y)
n.plot(X, y)
'''
'''
m = Model(2,2,[2], one_hot=True)
X, y = AND_dataset(one_hot=True)
m.train(X, y, epoch=500)
A, b = AND_dataset(one_hot=True)
print(m.raw_validate(A,b))
'''
'''
m = Model(3,3,[3])
X, y = input3output3(size=500)
val_loss, train_loss, params, test_X, test_y = error_vs_reg([3], input3output3, [0,0.5,1,3,5])
'''
m = Model(2,2,[12,12], one_hot=True)
n = Model(2,2,[12,12], one_hot=True)
X, y = pure_random(30)
#m.train(X, y, epoch=10000, alpha=0.001, optimizer='SGD')
n.train(X, y, epoch=25000, alpha=0.0005, optimizer='Adam')
#print(m.raw_validate(X, y))
print(n.raw_validate(X, y))
#m.plot(X, y, title='SGD')
n.plot(X, y, title='Adam')

# Note that default sklearn datasets are not one_hot
