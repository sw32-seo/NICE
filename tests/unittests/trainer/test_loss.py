import jax.numpy as jnp
import jax
from flax import nnx
from trainer.loss import nice_logistic_prior


class TestLoss:
    rng = jax.random.PRNGKey(42)

    def test_nice_logistic_prior(self):
        output = jnp.zeros((1, 784))
        scaling_factor = jnp.ones((784,))
        loss = nice_logistic_prior(output, scaling_factor)
        computed_loss = -jnp.log(1 + jnp.exp(output)) - jnp.log(1 + jnp.exp(-output))
        assert loss.shape == (1,)
        assert loss.dtype == jnp.float32
        assert loss == computed_loss.sum() + scaling_factor.sum()

        output = jnp.ones((16, 784))
        scaling_factor = jnp.ones((784,))
        loss = nice_logistic_prior(output, scaling_factor)
        assert loss.shape == (16,)
        assert loss.dtype == jnp.float32
