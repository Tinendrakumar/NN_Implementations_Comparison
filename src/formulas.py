# formulas.py

import math

def sig(x):
    # use logistic function as activation function
    # Sigmoid function = 1/(1+e^(-x))
    exp = math.e ** (-x)
    return 1 / (1.0 * (1 + exp))

def inv_sig(x):
    # derivative of the output of neuron with respect to its input
    return sig(x) * (1 - sig(x))

def err(o, t):
    # squared error function, o is the actual output value and t is the target output
    return 0.5 * (o - t) ** 2

def inv_err(o, t):
    # Derivative of squared error function with respect to o
    return o - t
