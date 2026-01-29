import jax
from core import Module, Initializer

module = Module()
initializer = Initializer()

class Linear(Module):
    """
    Linear (fully connected) layer.

    Args:
        in_features: Number of input features. (default: float32)
        out_features: Number of output features. (default: float32)
        bias: Whether to include bias. (default: True)
        weight_init: Initializer for weights. (default: xavier_uniform)
        bias_init: Initializer for bias. (default: zeros_init)
        key: JAX random key. (default: None, will be initialized with 0 if None) 
    """
    
    def __init__(self, in_features: float, out_features: float, 
                 bias: bool = True,
                 weight_init: Callable = initializer.xavier_uniform(),
                 bias_init: Callable = initializer.zeros_init(), 
                 key: int = 0):
        """
        Initialize the linear layer.

        Args:
            in_features: Number of input features. (default: float32)
            out_features: Number of output features. (default: float32)
            bias: Whether to include bias. (default: True)
            weight_init: Initializer for weights. (default: xavier_uniform)
            bias_init: Initializer for bias. (default: zeros_init)
        """
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.bias = bias
        self.weight_init = weight_init
        self.bias_init = bias_init
        self.key = jax.random.PRNGKey(key)
        
        # Parameters will be initialized on first call
        self.weight = None
        self.bias_param = None
    
    def __call__(self, x):
        """
        Forward pass.

        Args:
            x: Input tensor.

        Returns:
            Output tensor.
        """

        # Lazy initialization
        if self.weight is None:
            self.weight = self.weight_init(self.key, (self.in_features, self.out_features))
            if self.bias:
                self.bias_param = self.bias_init((self.out_features,))
        
        # Forward pass
        x = jnp.dot(x, self.weight)
        if self.bias and self.bias_param is not None:
            x = x + self.bias_param
        return x