import numpy as np
from optimizers.base import base

class AdamOpt(base):

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

    def optimize(self, mv1, mv2, t, alpha=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8, tolerance=0.0001):

        '''
        list(list(Num)), list(list(Num)), Int, Float, Float, Float, Float
            -> list(NP.ARRAY), list(NP.ARRAY)

        Requires:
        * t must be incremented by one after every iteration

        Optimizes and returns parameters W and b using Adam Optimizer
        Parameters:
        * mv1 - first moment vector
        * mv2 - second moment vector
        * t - timestep

        Returns:
        * parameter W
        * parameter b
        * mv1
        * mv2
        '''
        
        dW, db = self.get_gradient()

        for i in range(self.size+1):
            mv1[0][i] = beta1 * mv1[0][i] + (1-beta1) * dW[i]
            mv2[0][i] = beta2 * mv2[0][i] + (1-beta2) * np.square(dW[i])
            mv1W = mv1[0][i]/(1 - beta1 ** t)
            mv2W = mv2[0][i]/(1 - beta2 ** t)
            self.W[i] -= (alpha * mv1W)/(np.sqrt(mv2W) + epsilon)

            mv1[1][i] = beta1 * mv1[1][i] + (1-beta1) * db[i]
            mv2[1][i] = beta2 * mv2[1][i] + (1-beta2) * np.square(db[i])
            mv1b = mv1[1][i]/(1 - beta1 ** t)
            mv2b = mv2[1][i]/(1 - beta2 ** t)
            self.b[i] -= (alpha * mv1b)/(np.sqrt(mv2b) + epsilon)

        return self.W, self.b, mv1, mv2
