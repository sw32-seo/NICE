import jax
from flax import nnx
import jax.numpy as jnp
from trainer.loss import nice_logistic_prior


@nnx.jit
def train_step(graphdef, state, x):
    # merge the graphdef and state into a single object
    model, optimizer, metrics = nnx.merge(graphdef, state)

    def loss_fn(model):
        output, scaling_factor = model(x)
        log_likelihood = nice_logistic_prior(output, scaling_factor)
        return -log_likelihood.mean()  # take the negative of the loss to maximize the likelihood

    nll, grads = nnx.value_and_grad(loss_fn)(model)
    log_likelihood = -nll
    optimizer.update(model=model, grads=grads)
    metrics.update(likelihood=log_likelihood)
    state = nnx.state((model, optimizer, metrics))

    return log_likelihood, state


@nnx.jit
def validate_step(model, metrics, x):
    output, scaling_factor = model(x)
    log_likelihood = nice_logistic_prior(output, scaling_factor)
    metrics.update(likelihood=log_likelihood)
