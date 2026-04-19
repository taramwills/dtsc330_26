"""A neural network si a collection of layers. In this case, we're
calling it a multilayer perceptron because we're creating fully
connected neural networks."""

from typing import Iterator

from duq_ds3_2025.cuppajoe import layer, tensor


class MLP():
    def __init__(self, layers: list[layer.Layer]):
        self.layers = layers

    def forward(self, x: tensor.Tensor) -> tensor.Tensor:
        """Compute a forward pass through the entire neural net"""
        for layer in self.layers:
            x = layer.forward(x)
        return x
    
    def backward(self, grad: tensor.Tensor) -> tensor.Tensor:
        """The backward pass from a known gradient/error"""
        for layer in reversed(self.layers):
            grad = layer.backward(grad)
        return grad
    
    def params_and_grads(self) -> Iterator[tuple[tensor.Tensor, tensor.Tensor]]:
        """Return the weights and biases for every single layer in turn,
        along with their gradients"""
        for layer in self.layers:
            for pair in [(layer.w, layer.grad_w), (layer.b, layer.grad_b)]:
                yield pair

    def zero_parameters(self):
        """Set all weights and biases to 0"""
        for layer in self.layers:
            layer.grad_w[:] = 0
            layer.grad_b[:] = 0