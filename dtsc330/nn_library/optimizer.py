"""An optimizer updates the parameters of the layers based on the output
of the gradient."""

from duq_ds3_2025.cuppajoe import mlp


class Optimizer():
    def __init__(self, neural_network: mlp.MLP, learning_rate: float = 0.01):
        self.net = neural_network
        self.lr = learning_rate

    def step(self):
        """Take a step forward, backpropagate the error."""
        raise NotImplementedError
    

class SGD(Optimizer):
    """Stochastic gradient descent optimizer"""
    def step(self):
        for param, grad in self.net.params_and_grads():
            param -= grad * self.lr