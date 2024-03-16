import pickle as pkl  # noqa: S403
from pathlib import Path

import pytest

from rl_ai2thor.agents.agents import RandomAgent
from rl_ai2thor.agents.callbacks import BaseCallback
from rl_ai2thor.envs.ai2thor_envs import ITHOREnv

SEED = 0


# %% === Fixtures ===
@pytest.fixture()
def ithor_env():
    env = ITHOREnv()
    yield env
    env.close()


# %% === Test BaseAgent ===
NB_STEPS_1 = 10
NB_STEPS_2 = 20


class CountTriggeredCallback(BaseCallback):
    def __init__(self):
        self.nb_on_reset_triggered = 0
        self.nb_on_step_triggered = 0
        self.nb_on_close_triggered = 0

    def on_step(self, *args):  # noqa: ARG002
        self.nb_on_step_triggered += 1

    def on_reset(self):
        self.nb_on_reset_triggered += 1

    def on_close(self):
        self.nb_on_close_triggered += 1


def test_base_agent_callbacks(ithor_env):
    """Test that the callbacks are triggered."""
    callback = CountTriggeredCallback()
    agent = RandomAgent(env=ithor_env, callback=callback)
    assert callback.nb_on_reset_triggered == 0
    assert callback.nb_on_step_triggered == 0
    assert callback.nb_on_close_triggered == 0

    obs, _ = agent.reset()
    assert callback.nb_on_reset_triggered == 1

    _ = agent(obs)
    assert callback.nb_on_step_triggered == 0

    _, obs, _, _, _, _ = agent.step(obs)
    assert callback.nb_on_step_triggered == 1

    callback.nb_on_step_triggered = 0
    _, nb_steps, obs, _, _, _, _ = agent.continue_episode(obs, NB_STEPS_1)
    assert nb_steps == NB_STEPS_1
    assert callback.nb_on_step_triggered == nb_steps

    callback.nb_on_step_triggered = 0
    callback.nb_on_reset_triggered = 0
    _, total_nb_steps, obs, _, _, _, _ = agent.run_episode(1, total_max_steps=NB_STEPS_2)
    assert total_nb_steps == NB_STEPS_2
    assert callback.nb_on_step_triggered == total_nb_steps
    assert callback.nb_on_reset_triggered == 1

    agent.close()
    assert callback.nb_on_reset_triggered == 1
    assert callback.nb_on_step_triggered == total_nb_steps
    assert callback.nb_on_close_triggered == 1


# %% === Test random agent ===
def test_random_agent_same_runtime_reproducible(ithor_env):
    """Test that the random agent's choices are reproducible in the same runtime."""
    agent_1 = RandomAgent(env=ithor_env, seed=SEED)
    obs_1, _ = agent_1.reset()
    obs_2, _ = agent_1.reset()
    action_1_1 = agent_1(obs_1)
    action_1_2 = agent_1(obs_2)
    agent_2 = RandomAgent(env=ithor_env, seed=SEED)
    agent_2.reset()
    action_2_1 = agent_2(obs_1)
    action_2_2 = agent_2(obs_2)
    assert action_1_1 == action_2_1
    assert action_1_2 == action_2_2


def test_random_agent_same_runtime_non_reproducible(ithor_env):
    """Test that the random agent's choices are non-reproducible in the same runtime if not seeded."""
    agent_1 = RandomAgent(env=ithor_env)
    obs_1, _ = agent_1.reset()
    obs_2, _ = agent_1.reset()
    action_1_1 = agent_1(obs_1)
    action_1_2 = agent_1(obs_2)
    agent_2 = RandomAgent(env=ithor_env)
    agent_2.reset()
    action_2_1 = agent_2(obs_1)
    action_2_2 = agent_2(obs_2)
    assert action_1_1 != action_2_1 or action_1_2 != action_2_2


def test_random_agent_different_runtimes_reproducible(ithor_env):
    """Test that the random agent's choices are reproducible in different runtimes."""
    agent = RandomAgent(env=ithor_env, seed=SEED)
    obs_1, _ = agent.reset()
    obs_2, _ = agent.reset()
    action_1_1 = agent(obs_1)
    action_1_2 = agent(obs_2)

    data_path = Path("tests/data/test_random_agent_different_runtimes_reproducible_actions.pkl")
    to_serialize_data = (action_1_1, action_1_2)
    # with data_path.open("wb") as f:
    #     pkl.dump(to_serialize_data, f)
    with data_path.open("rb") as f:
        action_2_1, action_2_2 = pkl.load(f)  # noqa: S301

    assert action_1_1 == action_2_1
    assert action_1_2 == action_2_2
