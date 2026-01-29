from .module import Module
import jax
import jax.numpy as jnp
import math

class ReLU(Module):
    """
    Rectified Linear Unit activation function.
    """
    def __call__(self, x):
        return jnp.maximum(0, x)

class LeakyReLU(Module):
    """
    Leaky Rectified Linear Unit activation function.
    """
    def __init__(self, negative_slope: float = 0.01):
        super().__init__()
        self.negative_slope = negative_slope
    
    def __call__(self, x):
        return jnp.where(x >= 0, x, self.negative_slope * x)

class GELU(Module):
    """
    Gaussian Error Linear Unit activation function.
    """
    def __call__(self, x):
        return 0.5 * x * (1 + jax.lax.erf(x / math.sqrt(2)))

class Sigmoid(Module):
    """
    Sigmoid activation function.
    """
    def __call__(self, x):
        return 1 / (1 + jnp.exp(-x))

class Tanh(Module):
    """
    Hyperbolic tangent activation function.
    """
    def __call__(self, x):
        """
        Forward pass.

        Args:
            x (jax.numpy.ndarray): Input.

        Returns:
            jax.numpy.ndarray: Output.
        """
        return jnp.tanh(x)

class Softmax(Module):
    """
    Softmax activation function.
    """
    def __init__(self, dim: int = -1):
        """
        Initialize the softmax activation function.

        Args:
            dim (int): Dimension along which to compute softmax.
        """
        super().__init__()
        self.dim = dim
    
    def __call__(self, x):
        """
        Forward pass.
        Args:
            x (jax.numpy.ndarray): Input.
        Returns:
            jax.numpy.ndarray: Output.
        """
        return jax.nn.softmax(x, axis=self.dim)
