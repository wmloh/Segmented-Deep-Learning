import numpy as np

class base:
    
    '''
    The base object is the base optimizer object which simplifies
    initialization of functional optimizer objects.

    Note: This object only provides the get_gradient method and should
          not be used directly
    '''
    
    def __init__(self, W, b, delta, a, z, X, y, size, ddx):

        '''
        list(NP.ARRAY), list(NP.ARRAY), NP.ARRAY, list(NP.ARRAY),
            list(NP.ARRAY), NP.ARRAY, NP.ARRAY, Int, (Num -> Float)
                -> BASE

        Initializes the base object.
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
        self.W = W
        self.b = b
        self.delta = delta
        self.a = a
        self.z = z
        self.X = X
        self.y = y
        self.size = size
        self.ddx = ddx

    def get_gradient(self):

        '''
        None -> list(NP.ARRAY), list(NP.ARRAY)

        Returns the gradient of parameters W and b of the base object
        using the backpropagation algorithm.
        '''

        W = self.W
        delta = self.delta
        a = self.a
        z = self.z
        X = self.X
        y = self.y
        size = self.size
        ddx = self.ddx
   
        dW = list()
        db = list()

        delta[range(len(X)), y] -= 1
        delta = delta * self.ddx(z[-1])
        dW.append((a[-1].T).dot(delta))
        db.append(np.sum(delta, axis=0, keepdims=True))

        for l in range(size-1, 0, -1):
            delta = delta.dot(self.W[l+1].T) * self.ddx(z[l])
            dW.insert(0, (a[l-1].T).dot(delta))
            db.insert(0, np.sum(delta, axis=0, keepdims=True))

        delta = delta.dot(self.W[1].T) * self.ddx(z[0])
        dW.insert(0, (X.T).dot(delta))
        db.insert(0, np.sum(delta, axis=0))

        dW = np.array(dW)
        db = np.array(db)

        return dW, db
