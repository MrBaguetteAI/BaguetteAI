# BaguetteAI
An AI development library written in python.

## Install
    pip install BaguetteAI

## Example
    from BaguetteAI import *
    
    '''This is an example for a simple XOR Gate Neural Network'''
    
    TotalL = 3 #This is the total number of layers in the Network
    InputNeuronN = 2
    NumberHiddenN = 4 #This is the number of neurons that should be in each hidden layer
    NumberO = 1 #This is the number of the output layer neurons
    
    data = [([0, 0], [0]),
            ([0, 1], [1]),
            ([1, 0], [1]),
            ([1, 1], [0])
            ]
    
    W = neural.constructW(InputNeuronN, TotalL, NumberHiddenN, NumberO, randomize=True)
    print(W)
    NN = neural.constructL(TotalL, data[0][0], InputNeuronN, NumberHiddenN, NumberO, W)
    
    neural.training(NN, data, 10000, 0.5,1000,True)
    utils.save_json(neural.extract(NN), "XOR_weights.json")
    
    
    print(W)
