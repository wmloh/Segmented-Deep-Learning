import random
import numpy as np
from util.util import ctrlflow_onehot

def random_3inputs(size=500, one_hot=True):
    

    '''
    Int, Bool -> NP.ARRAY, NP.ARRAY

    Returns two np.arrays where the first is a set of 3 random
    inputs and the second is the labels which is 1 if and only
    if there are exactly 2 1's as inputs.
    '''
    
    output_X = list()
    output_y = list()
    setsize = int(size/8)
    t = ctrlflow_onehot(one_hot, 1, 2)
    f = ctrlflow_onehot(one_hot, 0, 2)
    output_X += [[[0,0,0],f] for i in range(setsize)]
    output_X += [[[0,0,1],f] for i in range(setsize)]
    output_X += [[[0,1,0],f] for i in range(setsize)]
    output_X += [[[0,1,1],t] for i in range(setsize)]
    output_X += [[[1,0,0],f] for i in range(setsize)]
    output_X += [[[1,0,1],t] for i in range(setsize)]
    output_X += [[[1,1,0],t] for i in range(setsize)]
    output_X += [[[1,1,1],f] for i in range(size - 7 * setsize)]
    
    random.shuffle(output_X)

    output_y = [x[1] for x in output_X]
    output_X = [x[0] for x in output_X]

    output_X = np.array(output_X)
    output_y = np.array(output_y, dtype='int32')
    return output_X, output_y

def input3output3(size=500, reduceSkew=True, one_hot=True):
    '''
    Int, Bool, Bool -> NP.ARRAY, NP.ARRAY

    Returns two np.arrays where the first is a set of 3 inputs 
    and the second is the labels which is:
    * 2 if there are three 1's
    * 1 if there are two 1's
    * 0 otherwise
    '''
    output_X = list()
    output_y = list()
    setsize = int(size/8)
    setsize_0 = size - 7 * setsize

    if reduceSkew:
        setsize_0 = size // 4
        setsize = (size - setsize_0) // 7
    c1 = ctrlflow_onehot(one_hot, 0, 3)
    c2 = ctrlflow_onehot(one_hot, 1, 3)
    c3 = ctrlflow_onehot(one_hot, 2, 3)
    output_X += [[[0,0,0],c1] for i in range(setsize)]
    output_X += [[[0,0,1],c1] for i in range(setsize)]
    output_X += [[[0,1,0],c1] for i in range(setsize)]
    output_X += [[[0,1,1],c2] for i in range(setsize)]
    output_X += [[[1,0,0],c1] for i in range(setsize)]
    output_X += [[[1,0,1],c2] for i in range(setsize)]
    output_X += [[[1,1,0],c2] for i in range(size - setsize * 6 - setsize_0)]
    output_X += [[[1,1,1],c3] for i in range(setsize_0)]
    
    random.shuffle(output_X)

    output_y = [x[1] for x in output_X]
    output_X = [x[0] for x in output_X]

    output_X = np.array(output_X)
    output_y = np.array(output_y, dtype='int32')
    return output_X, output_y

def pure_random(x, x_min=0, x_max=10, y_min=0, y_max=10, one_hot=True):
    
    '''
    Int, Num, Num, Num, Num, Bool -> NP.ARRAY, NP.ARRAY

    Returns two np.arrays where the first is a random set of 2
    inputs and the second is a set of random (unpredictable) labels
    '''

    t = ctrlflow_onehot(one_hot, 1, 2)
    f = ctrlflow_onehot(one_hot, 0, 2)
    
    return np.array([[random.uniform(x_min, x_max),
                      random.uniform(y_min, y_max)] for i in range(x)]),np.array([random.choice([t,f]) for i in range(x)])
