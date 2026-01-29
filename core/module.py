import jax
import jax.numpy as jnp
from jax import random, lax
from typing import Optional, Union, Tuple, List, Callable, Any
import functools
from dataclasses import dataclass
import math

class Module:
    """
    Base class for all neural network modules.

    Attributes:
        _params (dict): Dictionary of parameters.
        _submodules (dict): Dictionary of submodules.
        training (bool): Whether the module is in training mode.
    """
    def __init__(self):
        """
        Initialize the module.
        """
        self._params = {}
        self._submodules = {}
        self.training = True
        
    def __setattr__(self, name, value):
        """
        Set attribute.

        Args:
            name (str): Name of the attribute.
            value (Any): Value of the attribute.
        """

        if isinstance(value, Module):
            self._submodules[name] = value
        elif isinstance(value, (jnp.ndarray, Parameter)):
            if isinstance(value, Parameter):
                self._params[name] = value.value
            else:
                self._params[name] = value
        super().__setattr__(name, value)
    
    def parameters(self, prefix: str = ""):
        """
        Get all parameters as a flat dictionary.

        Args:
            prefix (str): Prefix for the parameter names.

        Returns:
            dict: Dictionary of parameters.
        """
        params = {}
        
        # Local parameters
        for name, param in self._params.items():
            key = f"{prefix}.{name}" if prefix else name
            params[key] = param
        
        # Submodule parameters
        for name, module in self._submodules.items():
            sub_prefix = f"{prefix}.{name}" if prefix else name
            params.update(module.parameters(sub_prefix))
        
        return params
    
    def named_modules(self, prefix: str = ""):
        """
        Get all modules with their names.

        Args:
            prefix (str): Prefix for the module names.

        Returns:
            list: List of modules with their names.
        """
        modules = [(prefix, self)] if prefix else [("", self)]
        
        for name, module in self._submodules.items():
            sub_prefix = f"{prefix}.{name}" if prefix else name
            modules.extend(module.named_modules(sub_prefix))
        
        return modules
    
    def init(self, key: jax.random.PRNGKey, input_shape: Tuple):
        """
        Initialize parameters with a dummy forward pass.

        Args:
            key (jax.random.PRNGKey): Random key for initialization.
            input_shape (Tuple): Shape of the input.

        Returns:
            Module: Initialized module.
        """
        dummy_input = random.normal(key, input_shape)
        _ = self.__call__(dummy_input)
        return self
    
    def train(self):
        """
        Set training mode.
        """
        self.training = True
        for module in self._submodules.values():
            module.train()
    
    def eval(self):
        """
        Set evaluation mode.
        """
        self.training = False
        for module in self._submodules.values():
            module.eval()
    
    def __call__(self, x):
        """
        Forward pass (to be implemented by subclasses).

        Args:
            x (jax.numpy.ndarray): Input.

        Returns:
            jax.numpy.ndarray: Output.
        """
        raise NotImplementedError

class Parameter:
    """
    Simple parameter wrapper.

    Args:
        value (jax.numpy.ndarray): Value of the parameter.
    """
    def __init__(self, value):
        self.value = value
    
    def __jax_array__(self):
        """
        Return the value of the parameter.

        Returns:
            jax.numpy.ndarray: Value of the parameter.
        """
        return self.value

class Normalizer(Module):
    """
    Base class for all normalizers.
    
    Args:
        normalized_shape: Shape of the normalized data.
        eps: Epsilon value for numerical stability.
        elementwise_affine: Whether to use elementwise affine transformation.
    """
    
    def __init__(self, normalized_shape, eps: float = 1e-5, elementwise_affine: bool = True):
        super().__init__()
        self.normalized_shape = normalized_shape if isinstance(normalized_shape, tuple) else (normalized_shape,)
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        
        if elementwise_affine:
            self.weight = jnp.ones(normalized_shape)
            self.bias = jnp.zeros(normalized_shape)
    
    def __call__(self, x):
        """
        Forward pass.

        Args:
            x (jax.numpy.ndarray): Input.

        Returns:
            jax.numpy.ndarray: Output.
        """

        mean = jnp.mean(x, axis=-1, keepdims=True)
        var = jnp.var(x, axis=-1, keepdims=True)
        
        x_norm = (x - mean) / jnp.sqrt(var + self.eps)
        
        if self.elementwise_affine:
            x_norm = x_norm * self.weight + self.bias
        
        return x_norm

class Dropout(Module):
    """Dropout layer."""
    
    def __init__(self, p: float = 0.5):
        super().__init__()
        self.p = p
    
    def __call__(self, x, key=None):
        if self.training and self.p > 0:
            if key is None:
                key = random.PRNGKey(0)
            keep_prob = 1 - self.p
            mask = random.bernoulli(key, p=keep_prob, shape=x.shape)
            return x * mask / keep_prob
        return x

class Sequential(Module):
    """Sequential container for stacking layers."""
    
    def __init__(self, *modules):
        super().__init__()
        for idx, module in enumerate(modules):
            self._submodules[f"layer_{idx}"] = module
    
    def __call__(self, x, **kwargs):
        for module in self._submodules.values():
            x = module(x, **kwargs)
        return x

class ModuleList(Module):
    """List of modules."""
    
    def __init__(self, modules=None):
        super().__init__()
        if modules:
            for idx, module in enumerate(modules):
                self._submodules[f"module_{idx}"] = module
    
    def append(self, module):
        idx = len(self._submodules)
        self._submodules[f"module_{idx}"] = module
    
    def __getitem__(self, idx):
        return list(self._submodules.values())[idx]
    
    def __len__(self):
        return len(self._submodules)
    
    def __call__(self, x):
        for module in self._submodules.values():
            x = module(x)
        return x

class ModuleDict(Module):
    """Dictionary of modules."""
    
    def __init__(self, modules=None):
        super().__init__()
        if modules:
            for name, module in modules.items():
                self._submodules[name] = module
    
    def __getitem__(self, key):
        return self._submodules[key]
    
    def __setitem__(self, key, module):
        self._submodules[key] = module
    
    def __call__(self, x):
        for module in self._submodules.values():
            x = module(x)
        return x
