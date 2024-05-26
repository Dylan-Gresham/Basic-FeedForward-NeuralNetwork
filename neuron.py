import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


class Neuron:
    """
    The Neuron class.

    Intended for use alongside the NeuralNetwork class.
    """

    def __init__(self, weights, bias):
        """Creates a new Neuron object with the specified weights and
        bias."""
        self.weights = weights
        self.bias = bias

    def feedforward(self, inputs):
        """Computes the weights of the inputs plus the bias
        then squishes it with the sigmoid function."""
        return sigmoid(np.dot(self.weights, inputs) + self.bias)
