import numpy as np
from nn_math import *
import datetime as dt
import matplotlib.pyplot as plt

class Model:
    '''
    A Model is a neural network capable of deep learning that can be part 
    of the Operation object
    '''
    def __init__(self, input_dim, output_dim, hl_dim, activation='tanh'):
        '''
        (Int, Int, list(Int), Str -> MODEL)
        Initializes a Model object with <input_dim> number of input values 
        and <output_dim> number of classes.
        <hl_dim> specifies the number of nodes in each layer and
        <activation> specifies the activation function of the neural network

        activation='tanh','sigmoid'
        '''
        self.input_dim = input_dim
        self.output_dim = output_dim
        if activation == 'tanh':
            self.activ = np.tanh
            self.ddx_activ = ddx_tanh
        elif activation == 'sigmoid':
            self.activ = expit
            self.ddx_activ = ddx_expit
        else:
            raise ValueError('The activation function specified does not exist')
        self.hl_dim = len(hl_dim)    
        self.W = list()
        self.b = list() 
        self.W.append(np.random.randn(input_dim, hl_dim[0]) / np.sqrt(input_dim))
        self.b.append(np.zeros((1, hl_dim[0])))

        for i in range(len(hl_dim) - 1):
        	self.W.append(np.random.randn(hl_dim[i], hl_dim[i + 1]) / np.sqrt(hl_dim[i]))
        	self.b.append(np.zeros((1, hl_dim[i+1])))

        self.W.append(np.random.randn(hl_dim[-1], output_dim) / np.sqrt(hl_dim[-1]))
        self.b.append(np.zeros((1, output_dim)))

    def show_params(self, showSize=False):
        '''
        Bool -> None
        Prints and returns the parameters of the Model, and dimension of 
        each parameter if showSize=True
        '''
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
    	return self.W, self.b

    def set_params(self, W, b):
        '''
        DEVELOPER TOOL
        list(NP.ARRAY), list(NP.ARRAY) -> None

        Manually configures the parameter of the model
        '''
        self.W = W
        self.b = b

    def train(self, X, y, epoch=5000,alpha=0.003):
        '''
        NP.ARRAY, NP.ARRAY, Int, Float -> None

        Trains the model with the given X=training_set and
        y=training_label with <epoch> iterations of the set
        and learning rate of <alpha>
        '''
        num_examples = len(X)
        afnc = self.activ
        for i in range(0, epoch):
        	z = list()
        	a = list()
        	z.append(X.dot(self.W[0]) + self.b[0])
        	for j in range(self.hl_dim):
        		a.append(afnc(z[j]))
        		z.append(a[j].dot(self.W[j+1]) + self.b[j+1])
            
        	exp_scores = np.exp(z[-1])    
        	probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

        	dW = list()
        	db = list()

        	delta = probs
        	delta[range(num_examples), y] -= 1
            
        	dW.append((a[-1].T).dot(delta))
        	db.append(np.sum(delta, axis=0, keepdims=True))

	        for l in range(self.hl_dim, 1, -1):
	        	delta = delta.dot(self.W[l].T) * self.ddx_activ(a[l-1])
	        	dW.insert(0, (a[l-2].T).dot(delta))
	        	db.insert(0, np.sum(delta, axis=0, keepdims=True))

	        delta = delta.dot(self.W[1].T) * self.ddx_activ(a[0])
	        dW.insert(0, (X.T).dot(delta))
	        db.insert(0, np.sum(delta, axis=0))

	        for j in range(self.hl_dim + 1):
	        	self.W[j] -= alpha * dW[j]
	        	self.b[j] -= alpha * db[j]

    def predict(self, x, display=True):
        '''
        NP.ARRAY, Bool -> INT

        Returns the prediction using the current parameters of the model.
        If display=True then prints the confidence of the returned prediction
        '''
        if type(x) is list:
            x = np.array(x)
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
        return np.argmax(probs, axis=1)

    def test(self, test_X, test_y):
        '''
        list(NP.ARRAY), list(NP.ARRAY) -> Float

        Predicts the test_X using the current parameters of the model and
        compares the predictions with the test_y and returns the percentage
        of the predictions that matches
        '''
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

    def plot(self, X, y, x_range=[], y_range=[], padding=1, delta=0.01):
        '''
        list(NP.ARRAY), list(NP.ARRAY), list(Int, Int), list(Int, Int), 
            Float, Float -> None

        Requires:
        * dim(X) = n x 2

        Plots the regression of the model while scatterplotting X=test_set
        with colour-coding using y=test_labels. Padding specifies the zoom
        magnitude and delta specifies the interval between each point predicted
        by the model.

        Note: If scatterplot of test set is not needed then pass X=[], y=[] but
              x_range and y_range must be satisfied.
        '''
        emptyDS = X == [] and y == []

        if not emptyDS:
            x_range = [X.min() - padding, X.max() + padding]
            y_range = [y.min() - padding, y.max() + padding]

        xx, yy = np.meshgrid(np.arange(x_range[0], x_range[1], delta), np.arange(y_range[0], y_range[1], delta))

        Z = self.predict(np.c_[xx.ravel(), yy.ravel()], display=False)
        Z = Z.reshape(xx.shape)

        plt.contourf(xx, yy, Z, cmap=plt.cm.winter)
        if not emptyDS:
            plt.scatter(X[:,0], X[:,1], c=y, cmap=plt.cm.spring)

        plt.show()
        

    
