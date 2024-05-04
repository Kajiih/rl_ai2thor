"""Integration tests for rl_thor package."""

import gymnasium as gym
import pytest

from rl_thor.agents.agents import RandomAgent
from rl_thor.envs.ai2thor_envs import ITHOREnv

# %% === Constants ===
MAX_STEPS = 200
NB_EPISODES = 5
random_agent_seed = 0


# %% === Fixtures ===
@pytest.fixture()
def ithor_env():
    env = ITHOREnv()
    yield env
    env.close()


@pytest.fixture()
def ithor_env_conf(config_override: dict):
    env = ITHOREnv(config_override=config_override)
    yield env
    env.close()


# %% === Test instantiating the environment with gym.make() ===
def test_gym_make() -> None:
    """Test instantiating the environment with gym.make()."""
    made_env: ITHOREnv = gym.make("rl_thor/ITHOREnv-v0.1")  # type: ignore
    expected_env = ITHOREnv()
    assert made_env.action_space == expected_env.action_space
    assert made_env.observation_space == expected_env.observation_space
    assert made_env.config == expected_env.config
    made_env.close()
    expected_env.close()


# %% === Running with random agent ===task_config_override = {
task_config_override = {
    "tasks": {"task_blueprints": [{"task_type": "PrepareMeal", "args": {}, "scenes": ["FloorPlan1"]}]}
}


@pytest.mark.parametrize("config_override", [task_config_override])
def test_random_agent_1_ep(ithor_env_conf) -> None:
    """Test running the environment with a random agent."""
    random_agent = RandomAgent(ithor_env_conf, seed=random_agent_seed)
    episode_output = random_agent.run_episode(nb_episodes=1, total_max_steps=MAX_STEPS)
    random_agent.close()
    _, total_nb_steps = episode_output[0], episode_output[1]

    assert total_nb_steps == MAX_STEPS


@pytest.mark.slow()
@pytest.mark.parametrize("config_override", [task_config_override])
def test_random_agent_n_ep(ithor_env_conf) -> None:
    """Test running the environment with a random agent."""
    random_agent = RandomAgent(ithor_env_conf, seed=random_agent_seed)
    run_output = random_agent.run_episode(nb_episodes=NB_EPISODES, total_max_steps=MAX_STEPS * NB_EPISODES)
    random_agent.close()
    _, total_nb_steps = run_output[0], run_output[1]

    assert total_nb_steps == MAX_STEPS * NB_EPISODES
