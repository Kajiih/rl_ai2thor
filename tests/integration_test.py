"""Integration tests for rl_ai2thor package."""

from rl_ai2thor.agents.agents import RandomAgent
from rl_ai2thor.envs.ai2thor_envs import ITHOREnv

# %% === Constants ===
MAX_STEPS = 200
SEED = 0
NB_EPISODES = 5


# %% === Running with random agent ===
def test_random_agent_1_ep() -> None:
    """Test running the environment with a random agent."""
    env = ITHOREnv()
    random_agent = RandomAgent(env)
    random_agent.run_episode(nb_episodes=1, total_max_steps=MAX_STEPS)
    random_agent.close()


def test_random_agent_n_ep() -> None:
    """Test running the environment with a random agent."""
    env = ITHOREnv()
    random_agent = RandomAgent(env)
    random_agent.run_episode(nb_episodes=NB_EPISODES, total_max_steps=MAX_STEPS * NB_EPISODES)
    random_agent.close()
