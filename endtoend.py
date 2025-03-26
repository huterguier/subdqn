import jax
import jax.numpy as jnp
import flax
import flax.linen as nn
import optax
import gymnax
import chex
import flashbax as fbx
from dataclasses import dataclass, field
from functools import partial


@jax.tree_util.register_dataclass
@chex.dataclass
class TrainState:
    args: dict = field(metadata=dict(static=True))
    params: flax.core.FrozenDict
    target: flax.core.FrozenDict
    opt_state: optax.OptState
    buffer_state: fbx.trajectory_buffer.TrajectoryBuffer
    state: gymnax.EnvState
    obs: chex.Array
    key: chex.Array
    step: int
    r: float


@chex.dataclass(frozen=True)
class TimeStep:
    obs: chex.Array
    action: chex.Array
    reward: chex.Array
    next_obs: chex.Array
    done: chex.Array


class DQN:
    def __init__(self, env, network, optimizer, buffer, **params):
        self.env, self.env_params = gymnax.make(env)
        self.network = network
        self.optimizer = optimizer
        self.buffer = buffer


    def init(self, key, args):
        state_key, network_key, optimizer_key, buffer_key = jax.random.split(key, 4)
        obs, state = self.env.reset(key)
        params = self.network.init(network_key, obs)
        target = self.network.init(network_key, obs)
        opt_state = self.optimizer.init(params)
        timestep = TimeStep(obs=obs, action=0, reward=0.0, done=False, next_obs=obs)
        buffer_state = self.buffer.init(timestep)
        v = TrainState(
            args=args,
            params=params,
            target=target,
            opt_state=opt_state,
            buffer_state=buffer_state,
            state=state,
            obs=obs,
            key=key,
            step=0,
            r=0.0,
        )

        return v

    def greedy_action(self, v, obs):
        q_values = self.network.apply(v.params, obs)
        action = jnp.argmax(q_values)
        return action

    def random_action(self, v, key):
        return jax.random.randint(key, (1,), 0, self.env.action_space(self.env_params).n)[0]

    def epsilon_greedy_action(self, v, key, obs, epsilon):
        key_exploration, key_actions = jax.random.split(key)
        return jax.lax.cond(
            jax.random.uniform(key_exploration, (1,))[0] < epsilon, 
            lambda v, key, obs: self.random_action(v, key_actions),
            lambda v, key, obs: self.greedy_action(v, obs),
            v,
            v.key,
            obs,
        )

    def update(self, v, batch):
        q_next_target = jax.vmap(self.network.apply, (None, 0))(v.target, batch.next_obs)
        q_next_target = jnp.max(q_next_target, axis=-1)
        next_q_value = batch.reward + (1 - batch.done) * 0.99 * q_next_target

        def loss(params):
            q_pred = jax.vmap(self.network.apply, (None, 0))(params, batch.obs)
            q_pred = q_pred[jnp.arange(q_pred.shape[0]), batch.action.squeeze()]
            return ((q_pred - next_q_value) ** 2).mean(), q_pred

        (loss_value, q_pred), grads = jax.value_and_grad(loss, has_aux=True)(v.params)
        updates, opt_state = self.optimizer.update(grads, v.opt_state)
        params = optax.apply_updates(v.params, updates)

        target = jax.lax.cond(
            v.step % v.args["target_update_frequency"] == 0,
            lambda params, target: optax.incremental_update(params, target, v.args["tau"]),
            lambda params, target: target,
            params,
            v.target,
        )

        v = v.replace(params=params, target=target, opt_state=opt_state)
        return v


    def step(self, v, action):
        obs, state = v.obs, v.state
        key, key_step = jax.random.split(v.key)
        next_obs, next_state, reward, done, _ = self.env.step(key_step, state, action, self.env_params)
        
        timestep=TimeStep(obs=obs, action=action, reward=reward, next_obs=next_obs, done=done) 
        buffer_state = self.buffer.add(v.buffer_state, timestep)

        # def print_yes(r):
        #     jax.debug.print("{r}", r=r)
        #     return 1.0
        #
        # def print_no(r):
        #     return 0.0
        # jax.lax.cond(
        #     done,
        #     print_yes,
        #     print_no,
        #     v.r + reward
        # )
        r = (1 - done) * (v.r + reward)
        
        v = v.replace(buffer_state=buffer_state, obs=next_obs, state=next_state, key=key, step=v.step+1, r=r)
        return v

    def linear_schedule(self, start_e, end_e, duration, t):
        slope = (end_e - start_e) / duration
        return jnp.maximum(slope * t + start_e, end_e)

    def learn(self, v, n_steps):

        def learn_step(v, key):
            epsilon = self.linear_schedule(v.args["start_e"], v.args["end_e"], v.args["exploration_fraction"] * v.args["total_timesteps"], v.step)
            action = self.epsilon_greedy_action(v, key, v.obs, epsilon)
            v = self.step(v, action)
            batch = self.buffer.sample(v.buffer_state, key).experience
            v = jax.lax.cond(v.step > v.args["training_starts"], self.update, lambda v, batch: v, v, batch)
            return v, None 

        keys = jax.random.split(v.key, v.args["total_timesteps"])
        v, _ = jax.lax.scan(learn_step, v, keys)

        return v


args = {
    "total_timesteps": 100000,
    "learning_rate": 2.5e-4,
    "gamma": 0.99,
    "tau": 1.0,
    "target_update_frequency": 500,
    "training_starts": 10000,
    "update_frequency": 10,
    "batch_size": 128,
    "start_e": 1.0,
    "end_e": 0.05,
    "learning_starts": 10000,
    "train_frequency": 1,
    "exploration_fraction": 0.5,
}

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



key = jax.random.PRNGKey(0)
env, env_params = gymnax.make("CartPole-v1")
network = QNetwork(action_dim=env.action_space(env_params).n)
optimizer = optax.adam(2.5e-4)
buffer = fbx.make_item_buffer(
    max_length=10000,
    min_length=128,
    sample_batch_size=128,
    add_sequences=False,
)
buffer = buffer.replace(
    init = jax.jit(buffer.init),
    add = jax.jit(buffer.add, donate_argnums=0),
    sample = jax.jit(buffer.sample),
    can_sample = jax.jit(buffer.can_sample),
)  

agent = DQN("CartPole-v1", network, optimizer, buffer)
# v = agent.init(key, args)

n_agents = 8
keys = jax.random.split(key, n_agents) 
vs = jax.vmap(agent.init, (0, None))(keys, args)

import time
start = time.time()
# v = agent.learn(v, key)
vs = jax.vmap(agent.learn, (0, 0))(vs, keys)
print(args["total_timesteps"] / (time.time() - start))
