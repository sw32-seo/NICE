import jax
import jax.numpy as jnp
from flax import nnx
from models.nice import NICE
import optax
from trainer.loss import nice_logistic_prior
from dataloader.mnist import get_mnist_dataloader
from trainer.train_step import train_step, validate_step
from tqdm import tqdm
import logging
import matplotlib.pyplot as plt


def main():
    model = NICE(dim_in=784, hidden_dim=1000, rngs=nnx.Rngs(jax.random.PRNGKey(42)))
    optimizer = nnx.Optimizer(model, optax.adam(1e-3), wrt=nnx.Param)
    metrics = nnx.MultiMetric(likelihood=nnx.metrics.Average("likelihood"),)

    train_dl, valid_dl, test_dl = get_mnist_dataloader(batch_size=512, seed=42)

    metrics_history = {
        "train_likelihood": [],
        "valid_likelihood": [],
        "test_likelihood": [],
    }
    for epoch in tqdm(range(2000)):
        graphdef, state = nnx.split((model, optimizer, metrics))
        for batch in train_dl:
            ll, state = train_step(graphdef, state, batch['image'])
        nnx.update((model, optimizer, metrics), state)

        metrics_history["train_likelihood"].append(metrics.compute()["likelihood"].item())
        metrics.reset()

        for batch in valid_dl:
            validate_step(model, metrics, batch['image'])
        metrics_history["valid_likelihood"].append(metrics.compute()["likelihood"].item())
        metrics.reset()

        for batch in test_dl:
            validate_step(model, metrics, batch['image'])
        metrics_history["test_likelihood"].append(metrics.compute()["likelihood"].item())
        metrics.reset()

        if (epoch + 1) % 100 == 0:
            logging.info(
                f"Epoch {epoch}, Train Likelihood: {metrics_history['train_likelihood'][-1]:.4f}, Valid Likelihood: {metrics_history['valid_likelihood'][-1]:.4f}, Test Likelihood: {metrics_history['test_likelihood'][-1]:.4f}"
            )
             # Draw samples
            z = jax.random.logistic(jax.random.PRNGKey(42), (100, 784))
            x = model.sampling(z)
            x = x.reshape(-1, 28, 28)
            plt.figure(figsize=(10, 10))
            for i in range(100):
                plt.subplot(10, 10, i + 1)
                plt.imshow(x[i].reshape(28, 28), cmap="gray")
                plt.axis("off")
            plt.savefig(f"samples_{epoch}_logistic.png")
            plt.close()

    # Draw training curves
    plt.figure(figsize=(10, 5))
    plt.plot(metrics_history["train_likelihood"], label="Train Likelihood")
    plt.plot(metrics_history["valid_likelihood"], label="Valid Likelihood")
    plt.plot(metrics_history["test_likelihood"], label="Test Likelihood")
    plt.legend()
    plt.savefig("training_curves_logistic.png")
    plt.close()


if __name__ == "__main__":
    main()
