import numpy as np
from optimizers.base import base

class SGD(base):

    '''
    The standard optimization algorithm object Stochastic Gradient
    Descent (SGD) for minimizing the cost function of a Model object.

    This class inherits from the base optimizer class.
    '''

    def __init__(self, W, b, delta, a, z, X, y, size, ddx):
        
        '''
        list(NP.ARRAY), list(NP.ARRAY), NP.ARRAY, list(NP.ARRAY),
            list(NP.ARRAY), NP.ARRAY, NP.ARRAY, Int, (Num -> Float)
                -> BASE

        Initializes the SGD object.
        Parameters:
        * W - weights list of arrays
        * b - biases list of arrays
        * delta - normalized expected scores
        * a - activated outputs list
        * z - outputs list
        * X - input dataset
        * y - labels for input dataset
        * size - number of layers for calculations
        * ddx - derivative of activation function
        '''
        super().__init__(W, b, delta, a, z, X, y, size, ddx)

    def optimize(self, alpha, reg_strength):

        '''
        Float, Float -> list(NP.ARRAY), list(NP.ARRAY)

        Optimizes and returns parameters W and b using SGD.
        Alpha is the learning rate and reg_strength is the
        regularization strength.
        '''
        
        dW, db = self.get_gradient()

        for i in range(self.size+1):
            dW[i] += reg_strength * self.W[i]
            self.W[i] -= alpha * dW[i]
            self.b[i] -= alpha * db[i]

        return self.W, self.b
