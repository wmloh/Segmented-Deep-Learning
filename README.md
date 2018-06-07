# Segmented Deep Neural Network
------------------------------------------------------------------------------------------------------------------------------------------
## Dependencies
* NumPy
* SciPy
* MatPlotLib
* Scikit-Learn

## Introduction

The idea of segmentation (not to be confused with Semantic Segmentation in Computer Vision) came about from experiences working with Convolutional Neural Networks (CNN). While increasing depth of neural networks, it becomes difficult to interpret the mechanisms behind the learning process. To avoid treating neural networks as a black box, a strategy is devised to provide control to developers as well understanding behind the learning process.

## Objects

There are two objects (as of June 6, 2018) in this repository: Model and Operation.

**Model** is a standard neural network object with fully customizable hidden layer architecture, capable of deep learning. The main limitation of this implementation is the inability to utilize GPU processing power or multiprocessing functions. Research and development is ongoing to improve training rate such as consider implementation with the _multiprocessing_ library in Python.

**Operation** is an object that arranges Model objects in a logical structure (a graph-like structure). Data inputs can be passed into the Operation object all at once; the first layer of Models will produce their predictions which is then passed to the second layer and the flow continues. An advantage of this is developers can assert control over the learning process through modularization of smaller individual Models. Debugging and retrain can potentially be simpler as a few small Models need to be retrain instead of the entire massive neural network. A disadvantage, however, is the requirement of feature engineering to manually define the architecture of the smaller neural network into a cohesive piece. Nevertheless, this object may prove useful in prototyping for domain experts despite some obvious limitations.

## Additional Tools

To assist constructing the neural network, some tools are provided.

_util.diagnosis_ enables training of multiple models with different regularization strength and automatically plots a graph of cost function with respect to regularization strength.

_datasets_ folder provides a simple dataset for quick training of a neural network. _datasets.logical_ provides AND, OR, NOT datasets and _datasets.random_ provides a few customized three inputs datasets and randomized datasets.

_optimizers_ folder provides two optimization modules: Stochastic Gradient Descent (default) and Adam Optimizer. 

## Sample Code

This sample code aims to produce a simple logic operation for classification:
_For a given A, B, C, D, the Operation object should predict based on this logic operation:
NOT ((A OR B) OR (C AND D))_

This code is written on ```main``` file:
```
from operation import Operation
from model import Model
from datasets.logical import *
from datasets.random import *
from util.diagnosis import *
from sklearn.datasets import *

X_and, y_and = AND_dataset(size=500)
X_or, y_or = OR_dataset(size=500)
X_not, y_not = NOT_dataset(size=500)

m1 = Model(2,2,[2], name='A OR B')
m2 = Model(2,2,[2], name='C AND D')
m3 = Model(2,2,[2], name='X OR Y')
m4 = Model(1,2,[2], name='NOT X')

m1.train(X_or, y_or, epoch=500, optimizer='Adam')
m2.train(X_and, y_and, epoch=500, optimizer='Adam')
m3.train(X_or, y_or, epoch=500, optimizer='Adam')
m4.train(X_not, y_not, epoch=500, optimizer='Adam')

Ops = Operation(4, 2, name='NOT ((A OR B) OR (C AND D))')
Ops.add(m1, 1)
Ops.add(m2, 1)
Ops.add(m3, 2)
Ops.add(m4, 3)

print(Ops)

test_set = [[1,0,0,1], [0,0,1,1], [1,1,0,1], [0,0,0,1]]

if Ops.check:
	for test in test_set:
		print(Ops(test))
```
Note: Predictions are given as one hot encoding: ```[1, 0]``` means class 0 (FALSE) and ```[0, 1]``` means class 1 (TRUE)

The architecture of the Models is relatively simple (1 hidden layer with 2 nodes). This is because elementary logic operations can be easily learned. For other more complex mathematical functions (e.g. sklearn.datasets.make_moons), more sophisticated hidden layer architecture may be needed.
