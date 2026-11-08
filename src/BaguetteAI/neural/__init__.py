from .neural_functions import (
    # matrix primitives
    Matrix_Multiply,
    Matrix_ADD,
    Matrix_Subtract,
    Matrix_Transpose,
    Matrix_Scale,
    Hadamard,
    # activation
    sigmoid,
    Sigmoid_Derivative,
    # backprop building blocks
    Backward_Output,
    Backward_Hidden,
    # network
    Network,
    Cache,
    Construct_Neural_Net,
    Forward,
    TrainStep,
    # demo
    main_run,
)

__all__ = [
    "Matrix_Multiply",
    "Matrix_ADD",
    "Matrix_Subtract",
    "Matrix_Transpose",
    "Matrix_Scale",
    "Hadamard",
    "sigmoid",
    "Sigmoid_Derivative",
    "Backward_Output",
    "Backward_Hidden",
    "Network",
    "Cache",
    "Construct_Neural_Net",
    "Forward",
    "TrainStep",
    "main_run",
]


#         from .neural import (
#     neuron,
#     constructW,
#     extract,
#     insert,
#     constructL,
#     compute,
#     backprop,
#     training_epochs,
#     training
# )

# __all__ = [
#         "neuron", "constructW", "extract", "insert", "constructL", "compute",
#     "backprop", "training_epochs", "training",
#     ]
