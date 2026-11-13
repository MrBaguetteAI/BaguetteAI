# BaguetteAI

An AI development library written in python.

## Install

    pip install BaguetteAI

## Example

    from BaguetteAI import *
    import random

    random.seed(0)   # if you expose a seed; otherwise C++ RNG still drives init

    # ---- build a 2 → 16 → 4 → 1 network ----
    net = neural.Construct_Neural_Net([2, 16, 4, 1])

    # XOR training set
    X = [
        [[0.0], [0.0]],
        [[0.0], [1.0]],
        [[1.0], [0.0]],
        [[1.0], [1.0]],
    ]
    Y = [
        [[0.0]],
        [[1.0]],
        [[1.0]],
        [[0.0]],
    ]

    LR = 0.5
    EPOCHS = 10000

    for epoch in range(EPOCHS):
        # shuffle so the order doesn't bias learning
        idx = list(range(4))
        random.shuffle(idx)

        total_loss = 0.0
        for i in idx:
            total_loss += neural.TrainStep(net, X[i], Y[i], LR)

        if epoch % 1000 == 0:
            print(f"epoch {epoch:5d}  loss {total_loss / 4:.6f}")

    # ---- final predictions ----
    print("\nXOR predictions:")
    for x, y in zip(X, Y):
        out = neural.Forward(net, x).a[-1][0][0]
        print(f"  {x[0][0]:.0f} XOR {x[1][0]:.0f}  →  {out:.4f}   (target {y[0][0]:.0f})")

    # ---- assert convergence ----
    ok = True
    for x, y in zip(X, Y):
        out = neural.Forward(net, x).a[-1][0][0]
        if abs(out - y[0][0]) > 0.3:
            ok = False
    print("\nPASS" if ok else "\nFAIL: network did not learn XOR")

    # ---- saving the weights and biases ----
    util.save_json("xor_weights.json", {
        "W": net.W,
        "b": net.b,
    })
