"""Run a random agent in the AI2Thor RL environment."""

# %%
import os
from pathlib import Path

from benchmark.experiment_utils import FullMetricsLogWrapper

from rl_thor.agents.agents import RandomAgent
from rl_thor.agents.callbacks import RecordVideoCallback
from rl_thor.envs.ai2thor_envs import ITHOREnv

SEED = 0

NB_EPISODES = 5
NB_STEPS = 1000

OUTPUT_DIR = Path("runs/random_agent/")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

run_idx = len(os.listdir(OUTPUT_DIR))
current_output_dir = OUTPUT_DIR / f"run_{run_idx}"
current_output_dir.mkdir(parents=True, exist_ok=True)


def main() -> None:
    """Run the environment with a random agent."""
    # Create the environment
    env = ITHOREnv()
    env = FullMetricsLogWrapper(env, log_dir=current_output_dir)
    env.reset(seed=SEED)
    callback = RecordVideoCallback(current_output_dir / "video.mp4")
    random_agent = RandomAgent(env, callback, seed=SEED)

    # Run the agent
    for _ in range(NB_EPISODES):
        obs, _info = random_agent.reset()
        print(env.task.text_description())
        random_agent.continue_episode(obs, max_steps=NB_STEPS)

    # Close the agent
    random_agent.close()

    print(f"Video recorded at {current_output_dir}/video.mp4")


if __name__ == "__main__":
    main()

# %%
