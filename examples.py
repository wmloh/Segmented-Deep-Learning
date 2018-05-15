import random
import numpy as np
import datetime as dt

def AND_dataset(size=500):
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
    '''
    output_X.shape = (size, 1)
    output_y.shape = (size, 1)
    '''
    return output_X, output_y


def random_3inputs(size=500):

    '''
    Outputs a 1 if and only if there are exactly 2 1's as inputs
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
    






    

    
