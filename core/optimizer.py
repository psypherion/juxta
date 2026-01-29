from dataclasses import dataclass
import jax.numpy as jnp

@dataclass
class OptimizerState:
    """Optimizer state container."""
    params: dict
    step: int = 0

class SGD:
    """
    Stochastic Gradient Descent optimizer.
    """

    def __init__(self, lr=0.01, momentum=0.0, weight_decay=0.0, **kwargs):
        """
        Initialize optimizer.
        """
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.kwargs = kwargs
    
    def init(self, params):
        """
        Initialize optimizer state.
        """
        velocities = {k: jnp.zeros_like(v) for k, v in params.items()}
        return OptimizerState(params=params, velocities=velocities)
    
    def update(self, state, grads):
        """
        Update optimizer state.
        """

        new_params = {}
        new_velocities = {}
        
        for key, param in state.params.items():
            grad = grads[key]
            
            # Weight decay
            if self.weight_decay > 0:
                grad = grad + self.weight_decay * param
            
            # Momentum
            if self.momentum > 0:
                velocity = state.velocities[key]
                velocity = self.momentum * velocity - self.lr * grad
                new_velocities[key] = velocity
                param_update = velocity
            else:
                param_update = -self.lr * grad
                
            new_params[key] = param + param_update
        
        return OptimizerState(
            params=new_params,
            velocities=new_velocities,
            step=state.step + 1
        )

class Adam:
    """
    Adam optimizer.
    """
    
    def __init__(self, lr=0.001, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0):
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.weight_decay = weight_decay
    
    def init(self, params):
        """
        Initialize optimizer state.
        """
        m = {k: jnp.zeros_like(v) for k, v in params.items()}
        v = {k: jnp.zeros_like(v) for k, v in params.items()}
        return OptimizerState(params=params, m=m, v=v)
    
    def update(self, state, grads):
        """
        Update optimizer state.
        """
        new_params = {}
        new_m = {}
        new_v = {}
        t = state.step + 1
        
        for key, param in state.params.items():
            grad = grads[key]
            
            # Weight decay
            if self.weight_decay > 0:
                grad = grad + self.weight_decay * param
            
            # Update biased first moment estimate
            m = self.beta1 * state.m[key] + (1 - self.beta1) * grad
            # Update biased second raw moment estimate
            v = self.beta2 * state.v[key] + (1 - self.beta2) * (grad ** 2)
            
            # Bias correction
            m_hat = m / (1 - self.beta1 ** t)
            v_hat = v / (1 - self.beta2 ** t)
            
            # Update parameters
            param_update = -self.lr * m_hat / (jnp.sqrt(v_hat) + self.eps)
            
            new_params[key] = param + param_update
            new_m[key] = m
            new_v[key] = v
        
        return OptimizerState(
            params=new_params,
            m=new_m, v=new_v,
            step=t
        )