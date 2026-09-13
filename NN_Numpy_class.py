import numpy as np
import pickle

class NNFromScratch:
    def __init__(self, lr:float, num_layers:list[int]):
        self.lr = lr
        self.numlayers = num_layers

        self.weights = [np.random.uniform(-1, 1, (num_layers[i], num_layers[i+1]))
                        for i in range(len(num_layers)-1)]
        self.bias = [np.zeros(num_layers[i])
                     for i in range(1, len(num_layers))]

        self.output = None


    

    def softmax(self, x):
        newx = x - np.max(x, axis=1, keepdims=True)
        exps = np.exp(newx)
        return exps / np.sum(exps, axis=1, keepdims=True)

    sigmoid = lambda self, ls: 1/(1+np.exp(-np.clip(ls, -50, 50)))
    sigmoid_deriv = lambda self, x: x*(1-x)

    def forward(self, current:np.array):

        activations = []
        inp_save = current
        
        lenght_layer = len(self.numlayers)-1

        for i in range(lenght_layer):
            z = current @ self.weights[i] + self.bias[i]
            
            if i == lenght_layer - 1:
                current = self.softmax(z)
            else:
                current = self.sigmoid(z)

            activations.append(current)

        self.output = (inp_save, activations)
        return current

    def calc_loss(self, labels):
        _, activations = self.output
        predictions = activations[-1]

        return -np.sum(labels * np.log(predictions + 1e-10)).item()
    
    
    def backward(self, labels:np.array):
        batch_size = self.output[0].shape[0]
        
        #CrossEntropyLoss_total = 0.

        x, activations = self.output
        predictions = activations[-1]

        CrossEntropyLoss_total = -np.sum(labels * np.log(predictions + 1e-10)).item()

        #Softmax + CrossEntropy derivative
        delta = predictions - labels

        #calc gradients
        weights_grad = [np.zeros_like(shape, shape=tuple([batch_size]+[n for n in shape.shape])) for shape in self.weights]
        bias_grad =    [np.zeros_like(shape, shape=tuple([batch_size]+[n for n in shape.shape])) for shape in self.bias]

        
        for i in reversed(range(len(self.weights))):
            if i==0:
                previous = x
            else:
                previous = activations[i-1]


            #dW/dL
            #weights_grad[i] += np.outer(previous, delta)
            weights_grad[i] += previous[:,:,None]*delta[:,None]

            #dB/dL
            bias_grad[i] += delta

            if i>0:
                delta = delta@self.weights[i].T
                #delta = self.weights[i] @ delta
                delta *= self.sigmoid_deriv(activations[i-1])


        #modify weights and biases and average gradients
        for i in range(len(self.weights)):
            weights_grad[i] = np.sum(weights_grad[i], axis=0) / batch_size
            bias_grad[i]    = np.sum(bias_grad[i]   , axis=0) / batch_size
        
        for i in range(len(self.weights)):
            self.weights[i] -= self.lr * weights_grad[i]
            self.bias[i] -= self.lr * bias_grad[i]

        self.output = []
        return CrossEntropyLoss_total / batch_size
    
    def train_batch(self, inputs, labels):
        self.forward(inputs)

        return self.backward(labels)

    def test_batch(self, inputs, labels):
        self.forward(inputs)

        return self.calc_loss(labels)

    def save_model(self, dirsave, bestloss):
        with open(dirsave, 'wb') as f:
            pickle.dump((self.weights, self.bias, self.numlayers, bestloss), f)

    def load_model(self, dirsave):
        with open(dirsave, 'rb') as f:
            self.weights, self.bias, self.numlayers, bestloss = pickle.load(f)

        return bestloss
