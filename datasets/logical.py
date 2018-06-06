import random
import numpy as np
from util.util import ctrlflow_onehot

def AND_dataset(size=500, one_hot=True):
    
    '''
    None -> NP.ARRAY, NP.ARRAY

    Returns two np.arrays where the first is a set of 2 random
    inputs (0 or 1) and the second is the labels when the inputs
    are passed into an AND operator.
    '''
    output_X = list()
    output_y = list()
    t = ctrlflow_onehot(one_hot, 1, 2)
    f = ctrlflow_onehot(one_hot, 0, 2)
    for i in range(size):
        roll = random.randint(0,3)
        if roll == 0:
            output_X.append([1,1])
            output_y.append(t)
        elif roll == 1:
            output_X.append([1,0])
            output_y.append(f)
        elif roll == 2:
            output_X.append([0,1])
            output_y.append(f)
        else:
            output_X.append([0,0])
            output_y.append(f)

    output_X = np.array(output_X)
    output_y = np.array(output_y, dtype='int32')
    
    return output_X, output_y
    
def OR_dataset(size=500, one_hot=True):
    
    '''
    None -> NP.ARRAY, NP.ARRAY

    Returns two np.arrays where the first is a set of 2 random
    inputs (0 or 1) and the second is the labels when the inputs
    are passed into an OR operator.
    '''
    output_X = list()
    output_y = list()
    t = ctrlflow_onehot(one_hot, 1, 2)
    f = ctrlflow_onehot(one_hot, 0, 2)
    for i in range(size):
        roll = random.randint(0,3)
        if roll == 0:
            output_X.append([1,1])
            output_y.append(t)
        elif roll == 1:
            output_X.append([1,0])
            output_y.append(t)
        elif roll == 2:
            output_X.append([0,1])
            output_y.append(t)
        else:
            output_X.append([0,0])
            output_y.append(f)

    output_X = np.array(output_X)
    output_y = np.array(output_y, dtype='int32')
    return output_X, output_y


def NOT_dataset(size=500, one_hot=True):
    
    '''
    None -> NP.ARRAY, NP.ARRAY

    Returns two np.arrays where the first is a set of random
    input (0 or 1) and the second is the labels when the input
    is passed into a NOT operator.
    '''
    output_X = list()
    output_y = list()
    t = ctrlflow_onehot(one_hot, 1, 2)
    f = ctrlflow_onehot(one_hot, 0, 2)
    for i in range(size):
        roll = random.randint(0,1)
        if roll == 0:
            output_X.append([0])
            output_y.append(t)
        else:
            output_X.append([1])
            output_y.append(f)

    output_X = np.array(output_X)
    output_y = np.array(output_y, dtype='int32')
    return output_X, output_y
