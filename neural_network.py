from neuron import Neuron
import numpy as np


class NeuralNetwork:
    """The Neural Network class."""

    def __init__(
            self,
            num_inputs,
            num_hidden_layers,
            num_outputs
    ):
        # Set given values
        self.num_inputs = num_inputs
        self.num_hidden_layers = num_hidden_layers
        self.num_outputs = num_outputs

        # Calculate random starting values for Neurons
        starting_bias = np.random.normal(loc=0.5, scale=0.5)
        hidden_layers = []
        for i in np.arange(num_hidden_layers):
            layer = []
            for j in np.arange(num_inputs):
                weights = np.random.normal(loc=0.0, scale=1.0,
                                           size=(1, num_inputs))
                layer.append(Neuron(weights, starting_bias))

            hidden_layers.append(layer)

        output_layer = []
        for i in np.arange(num_outputs):
            weights = np.random.normal(loc=0.0, scale=1.0,
                                       size=(1, num_inputs))
            output_layer.append(Neuron(weights, starting_bias))

        # Set Neuron layers
        self.hidden_layers = hidden_layers
        self.output_layer = output_layer

    def feedforward(self, inputs):
        if len(inputs) is not self.num_inputs:
            raise ValueError(f"The inputs are of invalid length. "
                             f"Expected length {self.num_inputs} but "
                             f"got length {len(inputs)}")

        prev_layer_outputs = inputs
        for layer in self.hidden_layers:
            current_layer_outputs = np.ndarray(shape=(1,
                                                      self.num_inputs),
                                               dtype=np.float64
                                               )
            for neuron in layer:
                current_layer_outputs = np.append(
                    current_layer_outputs,
                    neuron.feedforward(prev_layer_outputs)
                )
            prev_layer_outputs = current_layer_outputs

        outputs = np.ndarray(shape=(1, self.num_outputs),
                             dtype=np.float64)
        for neuron in self.output_layer:
            outputs = np.append(
                outputs,
                neuron.feedforward(prev_layer_outputs)
            )

        return np.mean(outputs)
