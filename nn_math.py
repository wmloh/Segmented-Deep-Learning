import numpy as np
from scipy.special import expit

def ddx_tanh(x):
	'''
	Num -> Float

	Returns the gradient of tanh curve at point <x>
	'''
    return (1 - np.power(x, 2))

def ddx_expit(x):
	'''
	Num -> Float

	Returns the gradient of sigmoid curve at point <x>
	'''
    return expit(x) * (1 - expit(x))
