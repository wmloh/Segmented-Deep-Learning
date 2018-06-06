import numpy as np
from scipy.special import expit

def ddx_tanh(x):
    
    '''
    Num -> Float

    Returns the gradient of tanh curve at point <x>
    '''
    return (1 - np.power(np.tanh(x), 2))

def ddx_expit(x):
    
    '''
    Num -> Float

    Returns the gradient of sigmoid curve at point <x>
    '''
    return expit(x) * (1 - expit(x))

def relu(x):
    
    '''
    Num -> Num

    Returns <x> where each element i in x is max(x[i],0)
    '''
    return np.maximum(x, 0, x)

def ddx_relu(x):
    
    '''
    Num -> Int

    Returns the gradient of relu curve at point <x>
    '''
    x[x < 0] = 0
    x[x >= 0] = 1
    return x
