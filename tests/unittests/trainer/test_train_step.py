import jax
from flax import nnx
import jax.numpy as jnp
from models.nice import NICE
from trainer.loss import nice_logistic_prior
import optax

from trainer.train_step import train_step


class TestTrainStep:

    def test_loss_reduction(self):
        model = NICE(dim_in=784, hidden_dim=1000, rngs=nnx.Rngs(jax.random.PRNGKey(42)))
        optimizer = nnx.Optimizer(model, optax.adam(1e-3), wrt=nnx.Param)
        metrics = nnx.MultiMetric(likelihood=nnx.metrics.Average("likelihood"),)
        x = jax.random.normal(jax.random.PRNGKey(42), (1, 784))

        # split the model, optimizer, and metrics into a graphdef and state
        graphdef, state = nnx.split((model, optimizer, metrics))

        initial_ll, state = train_step(graphdef, state, x)
        second_ll, state = train_step(graphdef, state, x)
        nnx.update((model, optimizer, metrics), state)
        assert (second_ll > initial_ll).item()
