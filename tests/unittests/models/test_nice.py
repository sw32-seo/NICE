import sys
import os
import jax
import jax.numpy as jnp
from flax import nnx

from models.nice import NICE, NICEBlock, MLP


class TestNICE:
    rng = jax.random.PRNGKey(42)

    def test_mlp(self):
        mlp = MLP(dim_in=28 * 28 // 2, hidden_dim=1000, rngs=nnx.Rngs(self.rng))
        x = jnp.ones((1, 28 * 28 // 2))
        y = mlp(x)
        assert y.shape == (1, 28 * 28 // 2)

    def test_nice_block(self):
        nice_block = NICEBlock(dim_in=28 * 28 // 2, hidden_dim=1000, rngs=nnx.Rngs(self.rng))

        x1 = jnp.ones((1, 28 * 28 // 2))
        x2 = jnp.ones((1, 28 * 28 // 2))
        y1, y2 = nice_block(x1, x2)
        assert y1.shape == (1, 28 * 28 // 2)
        assert y2.shape == (1, 28 * 28 // 2)

        # test the jit-compiled version
        jit_nice_block = jax.jit(nice_block)
        y1, y2 = jit_nice_block(x1, x2)
        assert y1.shape == (1, 28 * 28 // 2)
        assert y2.shape == (1, 28 * 28 // 2)

        # x1 and y1 should be the same
        assert jnp.allclose(x1, y1)
        # x2 and y2 should be different
        assert not jnp.allclose(x2, y2)

    def test_nice(self):
        nice = NICE(dim_in=28 * 28, hidden_dim=1000, rngs=nnx.Rngs(self.rng))
        x = jnp.ones((1, 28 * 28))
        y, s = nice(x)
        assert y.shape == (1, 28 * 28)
        assert s.shape == (28 * 28,)

        # test the jit-compiled version
        jit_nice = jax.jit(nice)
        y, s = jit_nice(x)
        assert y.shape == (1, 28 * 28)
        assert s.shape == (28 * 28,)

    def test_nice_sampling(self):
        nice = NICE(dim_in=28 * 28, hidden_dim=1000, rngs=nnx.Rngs(self.rng))
        z = jnp.ones((1, 28 * 28))
        x = nice.sampling(z)
        assert x.shape == (1, 28 * 28)

        # test the jit-compiled version
        jit_nice_sampling = jax.jit(nice.sampling)
        x = jit_nice_sampling(z)
        assert x.shape == (1, 28 * 28)
