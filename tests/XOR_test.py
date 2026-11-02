from baguetteai import *

'''This is an example for a simple XOR Gate Neural Network'''

TotalL = 3 #This is the total number of layers in the Network
NumberHiddenN = 4 #This is the number of neurons that should be in each hidden layer
NumberO = 1 #This is the number of the output layer neurons

data = [([0, 0], [0]),
        ([0, 1], [1]),
        ([1, 0], [1]),
        ([1, 1], [0])
        ]

W = neural.constructW(data[0][0], TotalL, NumberHiddenN, NumberO, randomize=True)
print(W)
NN = neural.constructL(TotalL, NumberHiddenN, NumberO, data[0][0], W)

neural.training(NN, data, 10000, 0.5,1000,True)
utils.save_json(neural.extract(NN), "XOR_weights.json")


print(W)
