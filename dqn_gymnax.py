# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/dqn/#dqn_jaxpy
import os
import random
import time
from dataclasses import dataclass
from functools import partial

import flax
import flax.core
import flax.linen as nn
import gymnax
from gymnax.wrappers import GymnaxToGymWrapper
import gymnasium as gym
import jax
import jax.numpy as jnp
import numpy as np
import optax
import tyro
import flashbax as fbx
import chex


@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    capture_video: bool = False
    """whether to capture videos of the agent performances (check out `videos` folder)"""
    save_model: bool = False
    """whether to save model into the `runs/{run_name}` folder"""

    # Algorithm specific arguments
    env_id: str = "CartPole-v1"
    """the id of the environment"""
    total_timesteps: int = 100000
    """total timesteps of the experiments"""
    learning_rate: float = 2.5e-4
    """the learning rate of the optimizer"""
    num_envs: int = 1
    """the number of parallel game environments"""
    buffer_size: int = 10000
    """the replay memory buffer size"""
    gamma: float = 0.99
    """the discount factor gamma"""
    tau: float = 1.0
    """the target network update rate"""
    target_network_frequency: int = 500
    """the timesteps it takes to update the target network"""
    batch_size: int = 128
    """the batch size of sample from the reply memory"""
    start_e: float = 1
    """the starting epsilon for exploration"""
    end_e: float = 0.05
    """the ending epsilon for exploration"""
    exploration_fraction: float = 0.5
    """the fraction of `total-timesteps` it takes from start-e to go end-e"""
    learning_starts: int = 10000
    """timestep to start learning"""
    train_frequency: int = 1
    """the frequency of training"""


def make_env(env_id, seed, idx, capture_video, run_name):
    def thunk():
        # env = gym.make(env_id)
        env, env_params = gymnax.make(env_id)
        env = GymnaxToGymWrapper(env, env_params) 
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env.action_space.seed(seed)

        return env

    return thunk


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

@chex.dataclass
class TrainState:
    params: flax.core.FrozenDict
    target_params: flax.core.FrozenDict
    opt_state: optax.OptState

# Define a simple tuple to hold the state of the environment. This is the format we will use to store transitions in our buffer.
@chex.dataclass(frozen=True)
class TimeStep:
    obs: chex.Array
    action: chex.Array
    done: chex.Array
    next_obs: chex.Array
    reward: chex.Array

def linear_schedule(start_e: float, end_e: float, duration: int, t: int):
    slope = (end_e - start_e) / duration
    return max(slope * t + start_e, end_e)


if __name__ == "__main__":
    

    args = tyro.cli(Args)
    f = open("dqn" + (str)(args.seed) + ".log", "w")
    assert args.num_envs == 1, "vectorized envs are not supported at the moment"
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    key = jax.random.PRNGKey(args.seed)
    key, q_key = jax.random.split(key, 2)

    # env setup
    envs = gym.vector.SyncVectorEnv(
        [make_env(args.env_id, args.seed + i, i, args.capture_video, run_name) for i in range(args.num_envs)]
    )
    assert isinstance(envs.single_action_space, gym.spaces.Discrete), "only discrete action space is supported"

    obs, _ = envs.reset(seed=args.seed)
    q_network = QNetwork(action_dim=envs.single_action_space.n)
    params = q_network.init(q_key, obs)
    tx = optax.adam(learning_rate=args.learning_rate)
    opt_state = tx.init(params)

    q_state = TrainState(
        params=params,
        target_params=params,
        opt_state=opt_state,
    )

    q_network.apply = jax.jit(q_network.apply)
    # This step is not necessary as init called on same observation and key will always lead to same initializations
    q_state = q_state.replace(target_params=optax.incremental_update(q_state.params, q_state.target_params, 1))

    buffer = fbx.make_item_buffer(
        max_length=args.buffer_size,
        min_length=args.batch_size,
        sample_batch_size=args.batch_size,
        add_sequences=False,
    )
    buffer = buffer.replace(
        init = jax.jit(buffer.init, backend='cpu'),
        add = jax.jit(buffer.add, donate_argnums=0, backend='cpu'),
        sample = jax.jit(buffer.sample, backend='cpu'),
        can_sample = jax.jit(buffer.can_sample, backend='cpu'),
    )  
    timestep = TimeStep(
        obs=obs, 
        action=jnp.array([envs.single_action_space.sample()]),
        reward= jnp.array([0.0]),
        done= np.array([False]),
        next_obs=obs,
    )
    buffer_state = buffer.init(timestep)

    @partial(jax.jit, donate_argnums=0)
    def update(q_state, observations, actions, next_observations, rewards, dones):
        q_next_target = q_network.apply(q_state.target_params, next_observations)  # (batch_size, num_actions)
        q_next_target = jnp.max(q_next_target, axis=-1)  # (batch_size,)
        next_q_value = rewards + (1 - dones) * args.gamma * q_next_target

        def mse_loss(params):
            q_pred = q_network.apply(params, observations)  # (batch_size, num_actions)
            q_pred = q_pred[jnp.arange(q_pred.shape[0]), actions.squeeze()]  # (batch_size,)
            return ((q_pred - next_q_value) ** 2).mean(), q_pred

        (loss_value, q_pred), grads = jax.value_and_grad(mse_loss, has_aux=True)(q_state.params)
        updates, q_state.opt_state = tx.update(grads, q_state.opt_state)
        q_state.params = optax.apply_updates(q_state.params, updates)

        return loss_value, q_pred, q_state

    start_time = time.time()

    # TRY NOT TO MODIFY: start the game
    obs, _ = envs.reset(seed=args.seed)
    for global_step in range(args.total_timesteps):
        # ALGO LOGIC: put action logic here
        epsilon = linear_schedule(args.start_e, args.end_e, args.exploration_fraction * args.total_timesteps, global_step)
        if random.random() < epsilon:
            actions = np.array([envs.single_action_space.sample() for _ in range(envs.num_envs)])
        else:
            q_values = q_network.apply(q_state.params, obs)
            actions = q_values.argmax(axis=-1)
            actions = jax.device_get(actions)

        # TRY NOT TO MODIFY: execute the game and log data.
        next_obs, rewards, terminations, truncations, infos = envs.step(actions)

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        if "final_info" in infos:
            for info in infos["final_info"]:
                if info and "episode" in info:
                    f.write(f"{global_step}, {info['episode']['r']}\n")
                    print(f"global_step={global_step}, episodic_return={info['episode']['r']}")

        # TRY NOT TO MODIFY: save data to reply buffer; handle `final_observation`
        real_next_obs = next_obs.copy()
        for idx, trunc in enumerate(truncations):
            if trunc:
                real_next_obs[idx] = infos["final_observation"][idx]

        timestep = TimeStep(
            obs=obs,
            action=actions,
            reward=rewards,
            done=terminations,
            next_obs=real_next_obs,
        )
        buffer_state = buffer.add(buffer_state, timestep)

        # TRY NOT TO MODIFY: CRUCIAL step easy to overlook
        obs = next_obs

        # ALGO LOGIC: training.
        if global_step > args.learning_starts:
            if global_step % args.train_frequency == 0:
                data = buffer.sample(buffer_state, q_key)
                batch = data.experience
                # perform a gradient-descent step
                loss, old_val, q_state = update(
                    q_state,
                    batch.obs.squeeze(1),
                    batch.action,
                    batch.next_obs.squeeze(1),
                    batch.reward.flatten(),
                    batch.done.flatten(),
                )

                if global_step % 100 == 0:
                    print("SPS:", int(global_step / (time.time() - start_time)))

            # update target network
            if global_step % args.target_network_frequency == 0:
                q_state = q_state.replace(
                    target_params=optax.incremental_update(q_state.params, q_state.target_params, args.tau)
                )

    envs.close()
    f.close()
