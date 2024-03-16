"""Integration tests for rl_ai2thor package."""

import gymnasium as gym
import pytest

from rl_ai2thor.agents.agents import RandomAgent
from rl_ai2thor.envs.ai2thor_envs import ITHOREnv

# %% === Constants ===
MAX_STEPS = 200
NB_EPISODES = 5
random_agent_seed = 0


# %% === Test instantiating the environment with gym.make() ===
def test_gym_make() -> None:
    """Test instantiating the environment with gym.make()."""
    made_env: ITHOREnv = gym.make("rl_ai2thor/ITHOREnv-v0.1")  # type: ignore
    expected_env = ITHOREnv()
    assert made_env.action_space == expected_env.action_space
    assert made_env.observation_space == expected_env.observation_space
    assert made_env.config == expected_env.config
    made_env.close()
    expected_env.close()


# %% === Running with random agent ===
def test_random_agent_1_ep() -> None:
    """Test running the environment with a random agent."""
    env = ITHOREnv()
    random_agent = RandomAgent(env, seed=random_agent_seed)
    episode_output = random_agent.run_episode(nb_episodes=1, total_max_steps=MAX_STEPS)
    random_agent.close()
    total_reward, total_nb_steps = episode_output[0], episode_output[1]

    assert total_nb_steps == MAX_STEPS
    # TODO: Use better assertions


@pytest.mark.slow()
def test_random_agent_n_ep() -> None:
    """Test running the environment with a random agent."""
    env = ITHOREnv()
    random_agent = RandomAgent(env, seed=random_agent_seed)
    try:
        run_output = random_agent.run_episode(nb_episodes=NB_EPISODES, total_max_steps=MAX_STEPS * NB_EPISODES)
    except TimeoutError as e:
        import pdb  # noqa PLC0415

        pdb.set_trace()
    random_agent.close()
    total_reward, total_nb_steps = run_output[0], run_output[1]

    assert total_nb_steps == MAX_STEPS * NB_EPISODES
