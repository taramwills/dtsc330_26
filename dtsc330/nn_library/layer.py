"""Layer of neurons that contains a set of tensors.
We will have to keep track of the ability to run and train."""

import numpy as np

from duq_ds3_2025.cuppajoe import tensor


class Layer():
    def __init__(self):
        self.w = tensor.Tensor
        self.b = tensor.Tensor
        self.x = None  # These are the inputs to the layer
        self.grad_w = 0
        self.grad_b = 0

    def forward(self, x: tensor.Tensor) -> tensor.Tensor:
        """A forward pass through the layer"""
        raise NotImplementedError
    
    def backward(self, x: tensor.Tensor) -> tensor.Tensor:
        """A training/backpropagation pass through the layer."""
        raise NotImplementedError


class Linear(Layer):
    def __init__(self, input_size: int, output_size: int):
        """Create a new linear layer"""
        super().__init__()
        self.w = np.random.randn(input_size, output_size)
        self.b = np.random.randn(output_size)

    def forward(self, x: tensor.Tensor) -> tensor.Tensor:
        """The forward computation is y = w x + b"""
        self.x = x
        return self.x @ self.w + self.b  # @ is multiplication of arrays
    
    def backward(self, grad: tensor.Tensor) -> tensor.Tensor:
        """We will compute the derivative on data passing backwards
        through the network to figure out the step we should take
        to train our network.
        
        We will compute the gradient for
        X = w*x + b
        y = f(x)
        dy/dw = f'(X)*x
        dy/dx = f'(X)*w
        dy/db = f'(x)

        The new component being added to our variables in tensor form:
        if y = f(x) and X = x @ w + b and f'(x) is the gradient then
        dy/dx = f'(X) @ w.T
        dy/dw = x.T @ f'(X)
        dy/db = f'(X)
        """
        self.grad_b = np.sum(grad, axis=0)
        self.grad_w = self.x.T @ grad
        return grad @ self.w.T


class Activation(Linear):
    """A generic activation layer type. Key is that it applies a
    function elementwise after computing x*w + b
    """
    def __init__(self, 
                 input_size: int, 
                 output_size: int,
                 f, 
                 f_prime) -> None:
        """Initialize an activation layer as a generic layer that also
        has a function and its derivative, from which the gradient can
        be computed

        Args:
            input_size (int): the number of input values to the layer
                (batch_size, input_size)
            output_size (int): the number of output values to the next
                layer (or final value)
                (batch_size, output_size)
            f (Callable[[tensor.Tensor], tensor.Tensor]): a 
                differentiable function
            f_prime (Callable[[tensor.Tensor], tensor.Tensor]): the 
                first derivative of f, as a function
        """
        super().__init__(input_size=input_size, output_size=output_size)
        self.f = f
        self.f_prime = f_prime

    def forward(self, x: tensor.Tensor) -> tensor.Tensor:
        self.x = x
        return self.f(super(Activation, self).forward(x))
    
    def backward(self, grad: tensor.Tensor) -> tensor.Tensor:
        """
        If y = f(x) and x = g(z)
        then dy/dz = f'(x) * g'(z)

        Think of f as the computation performed by the rest of the
        neural network and g being the component performed by this layer
        or as we called it in nn_by_scalar, the local derivative.
        """
        grad = super(Activation, self).backward(grad)
        return self.f_prime(self.x)*grad
    

def tanh(x: tensor.Tensor) -> tensor.Tensor:
    """Implement tanh function for tensors"""
    return np.tanh(x)

def tanh_prime(x: tensor.Tensor) -> tensor.Tensor:
    """First derivative of tanh"""
    y = tanh(x)
    return 1 - y**2


class Tanh(Activation):
    def __init__(self, input_size: int, output_size: int):
        super().__init__(input_size, output_size, tanh, tanh_prime)