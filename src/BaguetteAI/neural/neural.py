import math
import random

class neuron:
    __slots__ = ("inputs", "weights", "bias", "output", "delta")

    def __init__(self, inputs, weights, bias=0.0):
        self.inputs = inputs
        self.weights = weights
        self.bias = bias
        self.output = None
        self.delta = 0.0

    def summation(self):
        return sum(w * x for w, x in zip(self.weights, self.inputs)) + self.bias

    def activation(self,z):
        if z >= 0:
            return 1 / (1 + math.exp(-z))
        e = math.exp(z)
        return e / (1 + e)

    def forward(self):
        self.output = self.activation(self.summation())
        return self.output

    def backward(self, error):
        self.delta = error * self.output*(1-self.output)

    def update(self, LearningRate):
        for i, x in enumerate(self.inputs):
            self.weights[i] -= LearningRate*self.delta*x
        self.bias -= LearningRate*self.delta



def constructW(TotalL, Input, NumberHiddenN, NumberO, randomize=False):
    sizes = [Input] + [NumberHiddenN] * (TotalL - 2) + [NumberO]
    WM = []
    for n_in, n_out in zip(sizes[:-1], sizes[1:]):
        if randomize:
            WM.append([[random.uniform(-1, 1) for _ in range(n_in)] for _ in range(n_out)])
        else:
            row = [x**3 for x in range(1, n_in + 1)]
            WM.append([row[:] for _ in range(n_out)])
    return WM
        
def extract(nn):
    Weights = []
    Bias = []
    for layer in nn[1:]:
        tempW = []
        tempB = []
        for n in layer:
            tempW.append(n.weights)
            tempB.append(n.bias)
        Weights.append(tempW)
        Bias.append(tempB)
    return dict(weights = Weights, bias = Bias)

def insert(nn, data):
    """
    Load weights and biases from `extract()` output back into the network `nn`.
    `nn[0]` is the input layer (no weights), so we skip it just like extract does.
    """
    Weights = data["weights"]
    Bias    = data["bias"]

    for layer_idx, layer in enumerate(nn[1:]):
        for neuron_idx, neuron in enumerate(layer):
            neuron.weights = Weights[layer_idx][neuron_idx]
            neuron.bias    = Bias[layer_idx][neuron_idx]

    return nn

def constructL(L, I, H, O, W):
    nn = [[] for _ in range(L)]
    nn[0] = I
    for i in range(1, L):
        count = O if i == L-1 else H
        inputs = I if i == 1 else []   # filled in later for deeper layers
        nn[i] = [neuron(inputs, W[i-1][j]) for j in range(count)]
    return nn

def compute(nn):
    prev = nn[0]
    for layer in nn[1:]:
        for n in layer:
            n.inputs=prev
            n.forward()
        prev = [n.output for n in layer]
    return prev

def backprop(nn, targets, LearningRate):
    layers = nn[1:]

    for n,t in zip(layers[-1], targets):
        n.backward(n.output-t)

    for i in range(len(layers)-2, -1, -1):
        for j,n in enumerate(layers[i]):
            # Need to focus on this if dealing with different number of neurons in the hidden layers
            n.backward(sum(m.weights[j]*m.delta for m in layers[i+1]))

    for l in layers:
        for n in l:
            n.update(LearningRate)

def training_epochs(nn, dataSet, epochs, LR, report_every,show=True):
    for epoch in range(1,epochs+1):
        total = 0
        for x, target in dataSet:
            nn[0] = x
            out = compute(nn)
            total += 0.5 * sum((o - t) ** 2 for o, t in zip(out, target))
            backprop(nn, target, LR)
        if show:
            if epoch % report_every == 0:
                print(f"epoch {epoch}: loss {total / len(dataSet):.5f}")

def training(nn, dataSet, LR, show=True):
        total = 0
        for x, target in dataSet:
            nn[0] = x
            out = compute(nn)
            total += 0.5 * sum((o - t) ** 2 for o, t in zip(out, target))
            backprop(nn, target, LR)
