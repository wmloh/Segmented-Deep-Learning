import random
import numpy as np
import datetime as dt
from sklearn.datasets import *

def AND_dataset(size=500):
    '''
    None -> NP.ARRAY, NP.ARRAY

    Returns two np.arrays where the first is a set of 2 random
    inputs (0 or 1) and the second is the labels when the inputs
    are passed into an AND operator.
    '''
    output_X = list()
    output_y = list()
    for i in range(size):
        roll = random.randint(0,3)
        if roll == 0:
            output_X.append([1,1])
            output_y.append(1)
        elif roll == 1:
            output_X.append([1,0])
            output_y.append(0)
        elif roll == 2:
            output_X.append([0,1])
            output_y.append(0)
        else:
            output_X.append([0,0])
            output_y.append(0)

    output_X = np.array(output_X)
    output_y = np.array(output_y, dtype='int32')
    
    return output_X, output_y
    
def OR_dataset(size=500):
    '''
    None -> NP.ARRAY, NP.ARRAY

    Returns two np.arrays where the first is a set of 2 random
    inputs (0 or 1) and the second is the labels when the inputs
    are passed into an OR operator.
    '''
    output_X = list()
    output_y = list()
    for i in range(size):
        roll = random.randint(0,3)
        if roll == 0:
            output_X.append([1,1])
            output_y.append(1)
        elif roll == 1:
            output_X.append([1,0])
            output_y.append(1)
        elif roll == 2:
            output_X.append([0,1])
            output_y.append(1)
        else:
            output_X.append([0,0])
            output_y.append(0)

    output_X = np.array(output_X)
    output_y = np.array(output_y, dtype='int32')
    return output_X, output_y


def NOT_dataset(size=500):
    '''
    None -> NP.ARRAY, NP.ARRAY

    Returns two np.arrays where the first is a set of random
    input (0 or 1) and the second is the labels when the input
    is passed into a NOT operator.
    '''
    output_X = list()
    output_y = list()
    for i in range(size):
        roll = random.randint(0,1)
        if roll == 0:
            output_X.append([0])
            output_y.append(1)
        else:
            output_X.append([1])
            output_y.append(0)

    output_X = np.array(output_X)
    output_y = np.array(output_y, dtype='int32')
    return output_X, output_y


def random_3inputs(size=500):

    '''
    None -> NP.ARRAY, NP.ARRAY

    Returns two np.arrays where the first is a set of 3 random
    inputs and the second is the labels which is 1 if and only
    if there are exactly 2 1's as inputs.
    '''
    
    output_X = list()
    output_y = list()
    delta_size = int(size/8)
    output_X += [[[0,0,0],0] for i in range(delta_size)]
    output_X += [[[0,0,1],0] for i in range(delta_size)]
    output_X += [[[0,1,0],0] for i in range(delta_size)]
    output_X += [[[0,1,1],1] for i in range(delta_size)]
    output_X += [[[1,0,0],0] for i in range(delta_size)]
    output_X += [[[1,0,1],1] for i in range(delta_size)]
    output_X += [[[1,1,0],1] for i in range(delta_size)]
    output_X += [[[1,1,1],0] for i in range(size - 7 * delta_size)]
    
    random.shuffle(output_X)

    output_y = [x[1] for x in output_X]
    output_X = [x[0] for x in output_X]

    output_X = np.array(output_X)
    output_y = np.array(output_y, dtype='int32')
    return output_X, output_y
    
def pure_random(x, x_min=0, x_max=10, y_min=0, y_max=10):
    '''
    Int, Num, Num, Num, Num -> NP.ARRAY, NP.ARRAY

    Returns two np.arrays where the first is a random set of 2
    inputs and the second is a set of random (unpredictable) labels
    '''
    return np.array([[random.uniform(x_min, x_max),
                      random.uniform(y_min, y_max)] for i in range(x)]),np.array([random.randint(0,1) for i in range(x)])





    

    
