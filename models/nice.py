from flax import nnx
import jax.numpy as jnp


class MLP(nnx.Module):

    def __init__(self, dim_in: int, hidden_dim: int, rngs: nnx.Rngs):
        super().__init__()
        self.linear1 = nnx.Linear(dim_in, hidden_dim, rngs=rngs)
        self.linear2 = nnx.Linear(hidden_dim, hidden_dim, rngs=rngs)
        self.linear3 = nnx.Linear(hidden_dim, hidden_dim, rngs=rngs)
        self.linear4 = nnx.Linear(hidden_dim, hidden_dim, rngs=rngs)
        self.linear5 = nnx.Linear(hidden_dim, hidden_dim, rngs=rngs)
        self.linear6 = nnx.Linear(hidden_dim, dim_in, rngs=rngs)

    def __call__(self, x):
        x = self.linear1(x)
        x = nnx.relu(x)
        x = self.linear2(x)
        x = nnx.relu(x)
        x = self.linear3(x)
        x = nnx.relu(x)
        x = self.linear4(x)
        x = nnx.relu(x)
        x = self.linear5(x)
        x = nnx.relu(x)
        x = self.linear6(x)
        return x


class NICEBlock(nnx.Module):

    def __init__(self, dim_in: int, hidden_dim: int, rngs: nnx.Rngs):
        super().__init__()
        self.mlp = MLP(dim_in, hidden_dim, rngs)

    def __call__(self, x1, x2):
        y1 = x1
        y2 = x2 + self.mlp(x1)
        return y1, y2

    def sampling(self, z1, z2):
        x1 = z1
        x2 = z2 - self.mlp(z1)
        return x1, x2


class NICE(nnx.Module):

    def __init__(self, dim_in: int, hidden_dim: int, rngs: nnx.Rngs):
        super().__init__()
        dim_in_half = dim_in // 2
        self.block1 = NICEBlock(dim_in_half, hidden_dim, rngs)
        self.block2 = NICEBlock(dim_in_half, hidden_dim, rngs)
        self.block3 = NICEBlock(dim_in_half, hidden_dim, rngs)
        self.block4 = NICEBlock(dim_in_half, hidden_dim, rngs)
        self.scaling_factor = nnx.Param(jnp.ones((dim_in,)))

    def __call__(self, x):
        # split x into two halves by even and odd indices
        x1, x2 = x[:, ::2], x[:, 1::2]
        x1, x2 = self.block1(x1, x2)
        x2, x1 = self.block2(x2, x1)
        x1, x2 = self.block3(x1, x2)
        x2, x1 = self.block4(x2, x1)
        # concatenate x1 and x2 in even and odd indices
        output = jnp.zeros_like(x)
        output = output.at[:, ::2].set(x1)
        output = output.at[:, 1::2].set(x2)
        output = jnp.exp(self.scaling_factor.value) * output
        return output, self.scaling_factor.value

    def sampling(self, z):
        z = z / jnp.exp(self.scaling_factor.value)
        z1, z2 = z[:, ::2], z[:, 1::2]
        z2, z1 = self.block4.sampling(z2, z1)
        z1, z2 = self.block3.sampling(z1, z2)
        z2, z1 = self.block2.sampling(z2, z1)
        z1, z2 = self.block1.sampling(z1, z2)
        output = jnp.zeros_like(z)
        output = output.at[:, ::2].set(z1)
        output = output.at[:, 1::2].set(z2)
        return output
