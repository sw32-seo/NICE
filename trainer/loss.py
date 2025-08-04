import jax.numpy as jnp
import jax


def nice_logistic_prior(output, scaling_factor):
    # Use numerically stable softplus implementation
    log_ph_d = -jax.nn.softplus(output) - jax.nn.softplus(-output)
    loss = jnp.sum(log_ph_d, axis=1) + jnp.sum(scaling_factor)
    return loss
