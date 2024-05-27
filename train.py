import numpy as np
from neural_network import NeuralNetwork
from neuron import Neuron, sigmoid


def mse(y_true, y_pred):
    return ((y_pred - y_true) ** 2).mean()


class Trainer:
    def __init__(
            self,
            learning_rate,
            epochs,
            batch_size,
            tolerance
    ):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.tolerance = tolerance
        self.weights = None
        self.bias = None

    def predict(self, batch):
        # Predict based off the batch and current weights and bias
        return np.dot(batch, self.weights) + self.bias

    def gradient(self, batch, batch_expecteds):
        # Compute the prediction
        y_pred = self.predict(batch)

        # Compute the error
        error = y_pred - batch_expecteds

        # Compute the new weights and bias
        new_weights = np.dot(batch.T, error) / batch.shape[0]
        new_bias = np.mean(error)

        # Return them
        return new_weights, new_bias

    def fit(self, data, expecteds):
        # Determine the number of samples (rows) and the number of
        # features (columns)
        num_samples, num_features = data.shape

        # Randomly initialize the weights and biases
        self.weights = np.random.randn(num_features)
        self.bias = np.random.randn()

        # For each epoch
        for epoch in np.arange(self.epochs):
            # Randomly shuffle the dataset
            indices = np.random.permutation(num_samples)
            shuffled_data = data[indices]
            shuffled_expected = expecteds[indices]

            # For each batch of data
            for i in np.arange(0, num_samples, self.batch_size):
                # Get the batch of samples
                data_batch = shuffled_data[i:i + self.batch_size]
                # Get the corresponding batch of expecteds
                expected_batch = shuffled_expected[
                                 i:i + self.batch_size]

                # Compute the gradient weights and bias of the batch
                gradient_weights, gradient_bias = self.gradient(
                    data_batch,
                    expected_batch
                )

                # Update the weights
                self.weights -= self.learning_rate * gradient_weights
                # Update the bias
                self.bias -= self.learning_rate * gradient_bias

                # Every 10 epochs
                if epoch % 10 == 0:
                    # Compute the prediction
                    y_pred = self.predict(data)

                    # Compute the loss
                    loss = mse(expected_batch, y_pred)

                    # Print progress
                    print(f"Epoch: {epoch}, Loss: {loss}")

                # At the end of every epoch, check if we're below the
                # tolerance
                if np.linalg.norm(gradient_weights) < self.tolerance:
                    # If we are, print a message then break
                    print("Convergence has been reached!")
                    break

        # Return the weights and bias fitted to the input dataset
        return self.weights, self.bias

    def train(self, nn: NeuralNetwork, data, expecteds):
        # Calculate fitted weights and bias from the dataset
        weights, bias = self.fit(data, expecteds)

        # Construct a NeuralNetwork based off the fitted weights and
        # bias
        # TODO
