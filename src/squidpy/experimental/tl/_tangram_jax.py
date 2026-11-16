from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from jax.scipy.special import xlogy

__all__ = ["loss", "map_cells"]

# Adam's defaults, as in `torch.optim.Adam`
_BETA1, _BETA2, _EPS = 0.9, 0.999, 1e-8


def loss(logits: jax.Array, ref_x: jax.Array, query_x: jax.Array, density: jax.Array | None) -> jax.Array:
    """Tangram's `Mapper._loss_fn`, ``"cells"`` mode, default weights."""
    probs = jax.nn.softmax(logits, axis=1)
    pred = jnp.matmul(probs.T, ref_x, precision="highest")
    cosine = jnp.sum(pred * query_x, axis=0) / (jnp.linalg.norm(pred, axis=0) * jnp.linalg.norm(query_x, axis=0))
    total = -cosine.mean()
    if density is not None:
        # KL(density || mapped density); `xlogy` makes 0 * log(0) = 0, like `torch.nn.KLDivLoss`
        total += jnp.sum(xlogy(density, density) - density * jnp.log(probs.mean(axis=0)))
    return total


def map_cells(
    logits: np.ndarray,
    ref_x: np.ndarray,
    query_x: np.ndarray,
    density: np.ndarray | None,
    *,
    num_epochs: int,
    learning_rate: float,
) -> jax.Array:
    """Adam on `loss` from ``logits``; returns the ``(n_ref, n_query)`` mapping."""
    # bias corrections in float64, as torch computes them on the host
    steps = np.arange(1, num_epochs + 1)
    step_sizes = (learning_rate / (1 - _BETA1**steps)).astype(logits.dtype)
    second_corrections = np.sqrt(1 - _BETA2**steps).astype(logits.dtype)
    return _adam(logits, ref_x, query_x, density, step_sizes, second_corrections)


@partial(jax.jit, donate_argnums=0)
def _adam(logits, ref_x, query_x, density, step_sizes, second_corrections):  # noqa: PLR0917
    grad = jax.grad(loss)

    def step(state, corrections):
        logits, first, second = state
        step_size, second_correction = corrections
        g = grad(logits, ref_x, query_x, density)
        first = _BETA1 * first + (1 - _BETA1) * g
        second = _BETA2 * second + (1 - _BETA2) * g**2
        logits = logits - step_size * first / (jnp.sqrt(second) / second_correction + _EPS)
        return (logits, first, second), None

    zeros = jnp.zeros_like(logits)
    (logits, _, _), _ = jax.lax.scan(step, (logits, zeros, zeros), (step_sizes, second_corrections))
    return jax.nn.softmax(logits, axis=1)
