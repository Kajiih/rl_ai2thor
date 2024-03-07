"""Integration tests for rl_ai2thor package."""

import pytest

from rl_ai2thor.agents.agents import RandomAgent
from rl_ai2thor.envs.ai2thor_envs import ITHOREnv

# %% === Constants ===
MAX_STEPS = 200
NB_EPISODES = 5


# %% === Running with random agent ===
def test_random_agent_1_ep() -> None:
    """Test running the environment with a random agent."""
    env = ITHOREnv()
    random_agent = RandomAgent(env)
    episode_output = random_agent.run_episode(nb_episodes=1, total_max_steps=MAX_STEPS)
    random_agent.close()
    total_reward, total_nb_steps = episode_output[0], episode_output[1]

    assert total_nb_steps == MAX_STEPS
    # TODO: Use better assertions


@pytest.mark.slow()
def test_random_agent_n_ep() -> None:
    """Test running the environment with a random agent."""
    env = ITHOREnv()
    random_agent = RandomAgent(env)
    run_output = random_agent.run_episode(nb_episodes=NB_EPISODES, total_max_steps=MAX_STEPS * NB_EPISODES)
    random_agent.close()
    total_reward, total_nb_steps = run_output[0], run_output[1]

    assert total_nb_steps == MAX_STEPS * NB_EPISODES
