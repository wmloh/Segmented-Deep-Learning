import numpy as np
from scipy.special import expit
import datetime as dt

class Model:

    input_dim = 0
    output_dim = 0
    num_hlayer = 0
    activ = 'tanh'
    W1 = 0
    b1 = 0
    W2 = 0
    b2 = 0
    
    def __init__(self, input_dim, output_dim, num_hlayer, activation='tanh'):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_hlayer = num_hlayer
        self.activ = activation          
        
        self.W1 = np.random.randn(input_dim, num_hlayer) / np.sqrt(input_dim)
        self.b1 = np.zeros((1, num_hlayer))
        self.W2 = np.random.randn(num_hlayer, output_dim) / np.sqrt(num_hlayer)
        self.b2 = np.zeros((1, output_dim))

    def show_params(self):
        print("W1:")
        print(self.W1)
        print("b1:")
        print(self.b1)
        print("W2:")
        print(self.W2)
        print("b2:")
        print(self.b2)
        return self.W1, self.b1, self.W2, self.b2

    def _set_param(self, x, param):
        if type(x) is list:
            x = np.array(x)
        assert(type(x) is np.ndarray)
        if param == 'W1':
            self.W1 = x
        elif param == 'b1':
            self.b1 = x
        elif param == 'W2':
            self.W2 = x
        elif param == 'b2':
            self.b2 = x

    def train(self, X, y, epoch=5000,alpha=0.005,display=False, showRunTime=False):
        if showRunTime:
            time_1 = dt.datetime.now()
        num_examples = len(X)
        display_count = 0
        if self.activ == 'tanh':
            afnc = np.tanh
        elif self.activ == 'sigmoid':
            afnc = expit
        for i in range(0, epoch):
            z1 = X.dot(self.W1) + self.b1
            a1 = afnc(z1)
            z2 = a1.dot(self.W2) + self.b2
            
            exp_scores = np.exp(z2)
            
            probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

            delta3 = probs
            delta3[range(num_examples), y] -= 1
            
            if display:
                if display_count % 200 == 0:
                    print(delta3[:5])
                    print(z2[:5])
                display_count += 1
            
            dW2 = (a1.T).dot(delta3)
            db2 = np.sum(delta3, axis=0, keepdims=True)
            delta2 = delta3.dot(self.W2.T) * (1 - np.power(a1, 2))
            dW1 = np.dot(X.T, delta2)
            db1 = np.sum(delta2, axis=0)

            self.W1 -= alpha * dW1
            self.b1 -= alpha * db1
            self.W2 -= alpha * dW2
            self.b2 -= alpha * db2

        if showRunTime:
            print(dt.datetime.now() - time_1)

    def predict(self, x, display=True):
        if type(x) is list:
            x = np.array(x)
        z1 = x.dot(self.W1) + self.b1
        if self.activ == 'tanh':
            a1 = np.tanh(z1)
        elif self.activ == 'sigmoid':
            a1 = expit(z1)
        z2 = a1.dot(self.W2) + self.b2

        exp_scores = np.exp(z2)
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
            
        

    
