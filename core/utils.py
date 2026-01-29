from .module import Module

class Flatten(Module):
    """Flatten layer."""
    def __call__(self, x):
        """
        Forward pass.

        Args:
            x (jax.numpy.ndarray): Input.

        Returns:
            jax.numpy.ndarray: Output.
        """
        return x.reshape(x.shape[0], -1)

class Reshape(Module):
    """Reshape layer."""
    def __init__(self, shape):
        super().__init__()
        self.shape = shape
    
    def __call__(self, x):
        """
        Forward pass.

        Args:
            x (jax.numpy.ndarray): Input.

        Returns:
            jax.numpy.ndarray: Output.
        """
        return x.reshape((x.shape[0], *self.shape))

class Identity(Module):
    """Identity layer."""
    def __call__(self, x):
        """
        Forward pass.

        Args:
            x (jax.numpy.ndarray): Input.

        Returns:
            jax.numpy.ndarray: Output.
        """
        return x

class Lambda(Module):
    """Lambda layer for arbitrary operations."""
    def __init__(self, func):
        super().__init__()
        self.func = func
    
    def __call__(self, x):
        """
        Forward pass.

        Args:
            x (jax.numpy.ndarray): Input.

        Returns:
            jax.numpy.ndarray: Output.
        """
        return self.func(x)
