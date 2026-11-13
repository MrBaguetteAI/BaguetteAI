from BaguetteAI import *

net = neural.Construct(
    [(2, "linear"), (16, "sigmoid"), (4, "sigmoid"), (2, "sigmoid"), (1, "sigmoid")],
    "mse", seed=42,
)
X = [
    [0.0, 0.0], 
    [0.0, 1.0], 
    [1.0, 0.0], 
    [1.0, 1.0]
]

Y = [
    [0.0], 
    [1.0], 
    [1.0], 
    [0.0]
]   # float targets, shape (4, 1)

loss = neural.TrainLoop(net, X, [], Y, False,
                        epochs=10000, lr=0.5, report_every=1000)
for x, y in zip(X, Y):
    print(f"{x[0]} XOR {x[1]} -> {neural.Predict(net, x)[0]:.4f}  (target {y[0]})")
