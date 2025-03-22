import jax
import jax.numpy as jnp
import flax
from flax import linen as nn
import optax


# ALGO LOGIC: initialize agent here:
class QNetwork(nn.Module):
    action_dim: int

    @nn.compact
    def __call__(self, x: jnp.ndarray):
        x = nn.Dense(120)(x)
        x = nn.relu(x)
        x = nn.Dense(84)(x)
        x = nn.relu(x)
        x = nn.Dense(self.action_dim)(x)
        return x

@jax.jit
def cosine_similarity_optax(a, b):
    a_flat_params = flax.traverse_util.flatten_dict(a)
    a_flat_array = jnp.concatenate([p.flatten() for p in a_flat_params.values()])
    b_flat_params = flax.traverse_util.flatten_dict(b)
    b_flat_array = jnp.concatenate([p.flatten() for p in b_flat_params.values()])

    return optax.losses.cosine_similarity(a_flat_array, b_flat_array)

@jax.jit
def cosine_similarity_reduce(a, b):
    a_sums = jax.tree_util.tree_map(lambda x: jnp.sum(x**2), a)
    a_l2 = jnp.sqrt(jax.tree_util.tree_reduce(lambda x, y: x + y, a_sums))
    b_sums = jax.tree_util.tree_map(lambda x: jnp.sum(x**2), b)
    b_l2 = jnp.sqrt(jax.tree_util.tree_reduce(lambda x, y: x + y, b_sums))
    dot_sums = jax.tree_util.tree_map(lambda x, y: jnp.sum(x * y), a, b)
    dot = jax.tree_util.tree_reduce(lambda x, y: x + y, dot_sums)

    return dot / (a_l2 * b_l2)


key = jax.random.PRNGKey(2)
key1, key2 = jax.random.split(key)
obs = jnp.zeros((4,))
network = QNetwork(4)
params1 = network.init(key1, obs)
params2 = network.init(key2, obs)

import time

def measure(f):
    start = time.time()
    for _ in range(1000000):
        cs = f(params1, params2)
    print(time.time() - start)

measure(cosine_similarity_optax)
measure(cosine_similarity_reduce)
        

                                    
