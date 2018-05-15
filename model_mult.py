import numpy as np
from scipy.special import expit
import datetime as dt

class Model:

    input_dim = 0
    output_dim = 0
    hl_dim = 0
    activ = 'tanh'
    W = list()
    b = []
    
    def __init__(self, input_dim, output_dim, hl_dim, activation='tanh'):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.activ = activation     
        self.hl_dim = len(hl_dim)     
        self.W.append(np.random.randn(input_dim, hl_dim[0]) / np.sqrt(input_dim))
        self.b.append(np.zeros((1, hl_dim[0])))

        for i in range(len(hl_dim) - 1):
        	self.W.append(np.random.randn(hl_dim[i], hl_dim[i + 1]) / np.sqrt(hl_dim[i]))
        	self.b.append(np.zeros((1, hl_dim[i+1])))

        self.W.append(np.random.randn(hl_dim[-1], output_dim) / np.sqrt(hl_dim[-1]))
        self.b.append(np.zeros((1, output_dim)))

    def show_params(self):
    	for index, val in enumrate(W):
    		print("W%i: " % (index+1))
    		print(val)
    	for index, val in enumrate(b):
    		print("b%i: " % (index+1))
    		print(val)
    	return self.W, self.b

    def train(self, X, y, epoch=5000,alpha=0.005):
        num_examples = len(X)
        if self.activ == 'tanh':
            afnc = np.tanh
        elif self.activ == 'sigmoid':
            afnc = expit
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
            
        	# problem here
        	dW.append((a[-1].T).dot(delta))
        	db.append(np.sum(delta, axis=0, keepdims=True))

	        for l in range(self.hl_dim, 1, -1):
	        	delta = delta.dot(self.W[l].T) * (1 - np.power(a[l-1], 2))
	        	dW.insert(0, (a[l-2].T).dot(delta))
	        	db.insert(0, np.sum(delta, axis=0, keepdims=True))

	        delta = delta.dot(self.W[1].T) * (1 - np.power(a[0], 2))
	        dW.insert(0, (X.T).dot(delta))
	        db.insert(0, np.sum(delta, axis=0))

	        for j in range(self.hl_dim + 1):
	        	self.W[j] -= alpha * dW[j]
	        	self.b[j] -= alpha * db[j]

    def predict(self, x, display=True):
        if type(x) is list:
            x = np.array(x)
        if self.activ == 'tanh':
        	afnc = np.tanh
        elif self.activ == 'sigmoid':
        	afnc = expit
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
            return occurrence/total
            
        

    
