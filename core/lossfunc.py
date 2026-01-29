from jax import numpy as jnp
import jax


class LossFunc:
    def __init__(self, lossfunc: str, reduction: str = 'mean', **kwargs):
        self.lossfunc = lossfunc
        self.reduction = reduction
        self.kwargs = kwargs

    def __call__(self, pred, target):
        if self.lossfunc == 'mse':
            return self.mse_loss(pred, target, self.reduction, **self.kwargs)
        elif self.lossfunc == 'cross_entropy':
            return self.cross_entropy_loss(pred, target, self.reduction, **self.kwargs)
        elif self.lossfunc == 'binary_cross_entropy':
            return self.binary_cross_entropy(pred, target, self.reduction, **self.kwargs)
        elif self.lossfunc == 'l1':
            return self.l1_loss(pred, target, self.reduction, **self.kwargs)
        else:
            raise ValueError(f"Unknown loss function: {self.lossfunc}")

    def mse_loss(pred, target, reduction='mean', **kwargs):
        """Mean Squared Error loss."""
        loss = (pred - target) ** 2
        
        if reduction == 'mean':
            return jnp.mean(loss)
        elif reduction == 'sum':
            return jnp.sum(loss)
        elif reduction == 'none':
            return loss
        else:
            raise ValueError(f"Unknown reduction: {reduction}")

    def cross_entropy_loss(logits, labels, reduction='mean', **kwargs):
        """Cross Entropy loss."""
        # For one-hot labels
        if labels.ndim == logits.ndim:
            log_softmax = logits - jax.scipy.special.logsumexp(logits, axis=-1, keepdims=True)
            loss = -jnp.sum(labels * log_softmax, axis=-1)
        else:
            # For integer class labels
            loss = jax.nn.log_softmax(logits)[jnp.arange(logits.shape[0]), labels]
            loss = -loss
        
        if reduction == 'mean':
            return jnp.mean(loss)
        elif reduction == 'sum':
            return jnp.sum(loss)
        elif reduction == 'none':
            return loss
        else:
            raise ValueError(f"Unknown reduction: {reduction}")

    def binary_cross_entropy(pred, target, reduction='mean'):
        """Binary Cross Entropy loss."""
        epsilon = 1e-7
        pred = jnp.clip(pred, epsilon, 1 - epsilon)
        loss = -(target * jnp.log(pred) + (1 - target) * jnp.log(1 - pred))
        
        if reduction == 'mean':
            return jnp.mean(loss)
        elif reduction == 'sum':
            return jnp.sum(loss)
        elif reduction == 'none':
            return loss
        else:
            raise ValueError(f"Unknown reduction: {reduction}")

    def l1_loss(pred, target, reduction='mean'):
        """L1 loss (Mean Absolute Error)."""
        loss = jnp.abs(pred - target)
        
        if reduction == 'mean':
            return jnp.mean(loss)
        elif reduction == 'sum':
            return jnp.sum(loss)
        elif reduction == 'none':
            return loss
        else:
            raise ValueError(f"Unknown reduction: {reduction}")
