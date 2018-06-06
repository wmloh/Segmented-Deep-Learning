import numpy as np
from util.nn_math import *
from util.util import *
import datetime as dt
import matplotlib.pyplot as plt
import matplotlib.animation as ani
from matplotlib import style
from optimizers.SGD import SGD
from optimizers.AdamOpt import AdamOpt
from multiprocessing import Pool

class Model:
    
    '''
    A Model is a classification neural network object capable of
    deep learning that can be part of the Operation object
    '''

    names = list()
    
    def __init__(self, input_dim, output_dim, hl_dim, activation='tanh', one_hot=True, name=None):

        '''
        Int, Int, list(Int), Str, Bool, Str/None -> MODEL
        
        Initializes a Model object.
        Parameters:
        * input_dim - number of input data
        * output_dim - number of classes
        * hl_dim - number of nodes in each hidden layer (architecture)
        * activation - activation function (tanh, sigmoid, relu)
        * one_hot - format of predicted output
        * name - unique name of object
        '''
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.one_hot = one_hot
        if activation == 'tanh':
            self.activ = np.tanh
            self.ddx_activ = ddx_tanh
        elif activation == 'sigmoid':
            self.activ = expit
            self.ddx_activ = ddx_expit
        elif activation == 'relu':
            self.activ = relu
            self.ddx_activ = ddx_relu
        else:
            raise ValueError('The activation function specified does not exist')
        
        self.hl_dim = len(hl_dim)
        self._architecture = hl_dim
        self._activname = activation
        self.W = list()
        self.b = list() 
        self.W.append(np.random.randn(input_dim, hl_dim[0]) / np.sqrt(input_dim))
        self.b.append(np.zeros((1, hl_dim[0])))

        for i in range(len(hl_dim) - 1):
            self.W.append(np.random.randn(hl_dim[i], hl_dim[i + 1]) / np.sqrt(hl_dim[i]))
            self.b.append(np.zeros((1, hl_dim[i+1])))

        self.W.append(np.random.randn(hl_dim[-1], output_dim) / np.sqrt(hl_dim[-1]))
        self.b.append(np.zeros((1, output_dim)))
        if name == None:
            self.name = str(id(self))
        else:
            assert(type(name) is str)
            if name in self.names:
                raise NameError('This name already exists in another Model object')
            self.name = name
            self.names.append(name)

    def __str__(self):
        
        '''
        User-defined print function that prints the name/ID of the Model
        '''
        return 'MODEL: ' + self.name

    def __repr__(self):
        
        '''
        Returns the true string representation of the Model which is:
        * Input size
        * Number of classes
        * Hidden layer architecture
        * Activation function
        '''
        return "%s\nInput size: %i, Classes: %i, " % (self.__str__(),
                                                      self.input_dim,
                                                      self.output_dim,) + \
               "Hidden layer architecture: %s" % (self._architecture.__str__()) + \
               "\nActivation function: %s" % (self._activname)

    def __call__(self, x):
        
        '''
        The Model object is callable, i.e. it acts like a function to predict
        the class with the given <x> and parameters
        '''
        return self.predict(x, display=False)

    @property
    def p(self):

        '''
        Attribute of Model object that prints out the parameters in a clean
        format.
        Used as a debugging tool.        
        '''
        for index, val in enumerate(self.W):
            print("W%i: " % (index+1))
            print(val)

        for index, val in enumerate(self.b):
            print("b%i: " % (index+1))
            print(val)

    def params(self, display=True, showSize=False):

        '''
        Bool, Bool -> None
        
        Prints (if display=True)and returns the parameters of the Model,
        and dimension of each parameter if showSize=True
        '''
        if display:
            for index, val in enumerate(self.W):
                print("W%i: " % (index+1))
                print(val)
                if showSize:
                    print(val.shape)

            for index, val in enumerate(self.b):
                print("b%i: " % (index+1))
                print(val)
                if showSize:
                    print(val.shape)
            return
        return self.W, self.b

    def set_params(self, W, b):

        '''
        list(NP.ARRAY), list(NP.ARRAY) -> None

        Manually configures the parameter of the model
        '''
        self.W = W[:].copy()
        self.b = b[:].copy()

    def train(self, X, y, epoch=5000,alpha=0.003, reg_strength=0, optimizer='SGD', showLossCount=0):

        '''
        NP.ARRAY, NP.ARRAY, Int, Num, Num, Int -> None

        Trains the model with the given X=training_set and
        y=training_label with <epoch> iterations of the set
        and learning rate of <alpha> and regularization
        strength of <reg_strength>.

        Prints loss function value <showLossCount> number of
        time equally across the epoch.
        '''
        num_examples = len(X)
        afnc = self.activ
        showLoss = showLossCount != 0
        
        if showLossCount != 0:
            showInterval = epoch // showLossCount
            if showInterval <= 0:
                showInterval = 51
        # Potential error in line 190 & 193 when one_hot=False
        if optimizer == 'Adam':
            t = 1
            mv1 = [[0 for _ in range(self.hl_dim+1)] for _ in range(2)]
            mv2 = [[0 for _ in range(self.hl_dim+1)] for _ in range(2)]
            
        for i in range(epoch):
            z = list()
            a = list()
            z.append(X.dot(self.W[0]) + self.b[0])
            for j in range(self.hl_dim):
                a.append(afnc(z[j]))
                z.append(a[j].dot(self.W[j+1]) + self.b[j+1])
            
            exp_scores = np.exp(z[-1])    
            probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
            if optimizer == 'SGD':
                if self.one_hot:
                    opt = SGD(self.W, self.b, probs, a, z, X, from_one_hot(y), self.hl_dim, self.ddx_activ)
                else:
                    opt = SGD(self.W, self.b, probs, a, z, X, y, self.hl_dim, self.ddx_activ)
                self.W, self.b = opt.optimize(alpha, reg_strength)      
            elif optimizer == 'Adam':
                if self.one_hot:
                    opt = AdamOpt(self.W, self.b, probs, a, z, X, from_one_hot(y), self.hl_dim, self.ddx_activ)
                else:
                    opt = AdamOpt(self.W, self.b, probs, a, z, X, y, self.hl_dim, self.ddx_activ)
                self.W, self.b, mv1, mv2 = opt.optimize(mv1, mv2, t)
                t += 1
            else:
                raise ValueError('Invalid optimizer type entered.')
            
            if showLoss and i % showInterval == 0:
                print(self.validate(X, y, reg_strength=reg_strength))

    def predict(self, x, display=True, returnP=False, one_hot=None):

        '''
        NP.ARRAY, Bool, Bool -> INT

        Returns the prediction using the current parameters of the model.
        If display=True then prints the confidence of the returned prediction.
        If returnP=True then it returns the probability of the classes instead
        of the most likely class.
        '''
        if type(x) is list:
            x = np.array(x)
        if one_hot == None:
            one_hot = self.one_hot
        afnc = self.activ
        z = x.dot(self.W[0]) + self.b[0]
        
        for i in range(self.hl_dim):
            a = afnc(z)
            z = a.dot(self.W[i + 1]) + self.b[i + 1]

        exp_scores = np.exp(z)
        probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
        if display:
            for i in range(len(probs[0])):
                print("P(class_%i) = %f" % (i, probs[0][i]))
        if returnP:
            return probs
        elif one_hot:
            return to_one_hot(np.argmax(probs, axis=1), self.output_dim)
        else:
            return np.argmax(probs, axis=1)

    def raw_validate(self, test_X, test_y):

        '''
        NP.ARRAY, NP.ARRAY -> Float

        Predicts the test_X using the current parameters of the model and
        compares the predictions with the test_y and returns the percentage
        of the predictions that matches
        '''
        
        if self.one_hot:
            prediction = [self.predict(i, display=False) for i in test_X]
            result = np.array(prediction) == test_y
            
            result = [all(i) for i in result]
            unique, count = np.unique(result, return_counts=True)
            if unique.shape == (1,):
                if unique[0]:
                    return 1.00
                else:
                    return 0.00
            else:
                lst = list(unique)
                index = lst.index(True)
                occurrence = count[index]
                total = np.sum(count)
                return round(occurrence/total, 5)
        else:
            result = self.predict(test_X, display=False) == test_y
            unique, count = np.unique(result, return_counts=True)
            
            if unique.shape == (1,):
                if unique[0]:
                    return 1.00
                else:
                    return 0.00
            else:
                lst = list(unique)
                index = lst.index(True)
                occurrence = count[index]
                total = np.sum(count)
                return round(occurrence/total, 5)

    def validate(self, X, y, reg_strength=0):

        '''
        NP.ARRAY, NP.ARRAY, Num -> Float

        Returns the categorical cross-entropy loss function of predictions
        on X and comparing them with y
        '''
        z = X.dot(self.W[0]) + self.b[0]
        afnc = self.activ
        for j in range(self.hl_dim):
            a = afnc(z)
            z = a.dot(self.W[j+1]) + self.b[j+1]
        
        exp_scores = np.exp(z)    
        probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

        if self.output_dim != 2:
            y = np.array([list(x).index(1) for x in y])

        logprobs = -np.log(probs[range(len(X)), y])
        data_loss = np.sum(logprobs)
        if reg_strength != 0:
            for i in range(self.hl_dim + 1):
                data_loss += reg_strength/2 * (np.sum(np.square(self.W[i])))

        return 1./ len(X) * data_loss

    
    def plot(self, X, y, x_range=[], y_range=[], padding=1, delta=0.01, title=None):

        '''
        NP.ARRAY, NP.ARRAY, list(Int, Int), list(Int, Int), 
            Float, Float, Str/None -> None

        Requires:
        * dim(X) = n x 2

        Plots the regression of the model while scatterplotting X=test_set
        with colour-coding using y=test_labels. Padding specifies the zoom
        magnitude and delta specifies the interval between each point predicted
        by the model.

        Note: If scatterplot of test set is not needed then pass X=[], y=[] but
              x_range and y_range must be given.
        '''
        
        emptyDS = X == []
        if not emptyDS:
            assert(X.shape[1] == 2)
            
        fig = plt.figure()
        ax1 = fig.add_subplot(111)
        if title != None:
            fig.canvas.set_window_title(title)
        
        if not emptyDS:
            x_range = [X[:,0].min() - padding, X[:,0].max() + padding]
            y_range = [X[:,1].min() - padding, X[:,1].max() + padding]

        xx, yy = np.meshgrid(np.arange(x_range[0], x_range[1], delta), np.arange(y_range[0], y_range[1], delta))

        Z = self.predict(np.c_[xx.ravel(), yy.ravel()], display=False, one_hot=False)
        Z = Z.reshape(xx.shape)

        if self.one_hot:
            y = from_one_hot(y)
        ax1.contourf(xx, yy, Z, cmap=plt.cm.winter)
        if not emptyDS:
            ax1.scatter(X[:,0], X[:,1], c=y, cmap=plt.cm.spring)

        fig.show()
        
        

    
