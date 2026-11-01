# BaguetteAI
An AI development library written in python.

## Install
    pip install baguetteai

## Example
    from baguetteai import constructW, constructL, training, compute

    data = [([0,0],[0]), ([0,1],[1]), ([1,0],[1]), ([1,1],[0])]
    W = constructW(data[0][0], 3, 4, 1, randomize=True)
    NN = constructL(3, 4, 1, data[0][0], W)
    training(NN, data, 10000, 0.5)
