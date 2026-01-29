"""
Core module initialization.
"""

__version__ = "0.1.0"

from .module import Module, Parameter, Normalizer, Dropout

class Initializer:
    """
    Base class for all initializers.
    """
    def __init__(self, initializer: str = "zeros_init", **kwargs):
        self.initializer = initializer
        self.kwargs = kwargs
    
    def __call__(self, key, shape):
        if self.initializer == "zeros_init":
            return self.zeros_init(shape, **self.kwargs)
        elif self.initializer == "ones_init":
            return self.ones_init(shape, **self.kwargs)
        elif self.initializer == "uniform_init":
            return self.uniform_init(scale, **self.kwargs)
        elif self.initializer == "normal_init":
            return self.normal_init(stddev, **self.kwargs)
        elif self.initializer == "xavier_uniform":
            return self.xavier_uniform(gain, **self.kwargs)
        elif self.initializer == "xavier_normal":
            return self.xavier_normal(gain, **self.kwargs)
        elif self.initializer == "kaiming_uniform":
            return self.kaiming_uniform(mode, nonlinearity='relu', **self.kwargs)
        elif self.initializer == "kaiming_normal":
            return self.kaiming_normal(mode, nonlinearity='relu', **self.kwargs)
        else:
            raise ValueError(f"Unknown initializer: {self.initializer}")

    def zeros_init(shape, dtype=jnp.float32):
        """Initialize with zeros."""
        return jnp.zeros(shape, dtype=dtype)

    def ones_init(shape, dtype=jnp.float32):
        """Initialize with ones."""
        return jnp.ones(shape, dtype=dtype)

    def uniform_init(scale=0.05, dtype=jnp.float32):
        """Uniform initialization."""
        def init(key, shape):
            return random.uniform(key, shape, dtype=dtype, minval=-scale, maxval=scale)
        return init

    def normal_init(stddev=0.01, dtype=jnp.float32):
        """Normal initialization."""
        def init(key, shape):
            return random.normal(key, shape, dtype=dtype) * stddev
        return init

    def xavier_uniform(gain=1.0, dtype=jnp.float32):
        """Xavier uniform initialization (also called Glorot uniform)."""
        def init(key, shape):
            if len(shape) < 2:
                raise ValueError("Xavier initialization requires at least 2 dimensions")
            
            fan_in, fan_out = shape[-2], shape[-1]
            limit = gain * math.sqrt(6.0 / (fan_in + fan_out))
            return random.uniform(key, shape, dtype=dtype, minval=-limit, maxval=limit)
        return init

    def xavier_normal(gain=1.0, dtype=jnp.float32):
        """Xavier normal initialization (also called Glorot normal)."""
        def init(key, shape):
            if len(shape) < 2:
                raise ValueError("Xavier initialization requires at least 2 dimensions")
            
            fan_in, fan_out = shape[-2], shape[-1]
            std = gain * math.sqrt(2.0 / (fan_in + fan_out))
            return random.normal(key, shape, dtype=dtype) * std
        return init

    def kaiming_uniform(mode='fan_in', nonlinearity='relu', dtype=jnp.float32):
        """Kaiming (He) uniform initialization."""
        def init(key, shape):
            if len(shape) < 2:
                raise ValueError("Kaiming initialization requires at least 2 dimensions")
            
            fan_in = shape[-2] if mode == 'fan_in' else shape[-1]
            gain = math.sqrt(2.0) if nonlinearity == 'relu' else 1.0
            bound = gain * math.sqrt(3.0 / fan_in)
            return random.uniform(key, shape, dtype=dtype, minval=-bound, maxval=bound)
        return init

    def kaiming_normal(mode='fan_in', nonlinearity='relu', dtype=jnp.float32):
        """Kaiming (He) normal initialization."""
        def init(key, shape):
            if len(shape) < 2:
                raise ValueError("Kaiming initialization requires at least 2 dimensions")
            
            fan_in = shape[-2] if mode == 'fan_in' else shape[-1]
            gain = math.sqrt(2.0) if nonlinearity == 'relu' else 1.0
            std = gain / math.sqrt(fan_in)
            return random.normal(key, shape, dtype=dtype) * std
        return init
