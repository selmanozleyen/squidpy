from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp
import optax
from jax.scipy.special import xlogy

__all__ = ["loss", "map_cells"]


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


@partial(jax.jit, static_argnames="num_epochs", donate_argnums=0)
def map_cells(
    logits: jax.Array,
    ref_x: jax.Array,
    query_x: jax.Array,
    density: jax.Array | None,
    *,
    num_epochs: int,
    learning_rate: float,
) -> jax.Array:
    """Adam, with `torch.optim.Adam`'s defaults, on `loss` from ``logits``; returns the ``(n_ref, n_query)`` mapping."""
    optimizer = optax.adam(learning_rate)
    grad = jax.grad(loss)

    def step(state, _):
        logits, opt_state = state
        updates, opt_state = optimizer.update(grad(logits, ref_x, query_x, density), opt_state)
        return (optax.apply_updates(logits, updates), opt_state), None

    (logits, _), _ = jax.lax.scan(step, (logits, optimizer.init(logits)), length=num_epochs)
    return jax.nn.softmax(logits, axis=1)
