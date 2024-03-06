"""Run a random agent in the AI2Thor RL environment."""

# %%
import os

from rl_ai2thor.agents.agents import RandomAgent
from rl_ai2thor.agents.callbacks import RecordVideoCallback
from rl_ai2thor.envs.ai2thor_envs import ITHOREnv
from rl_ai2thor.utils.general_utils import ROOT_DIR

SEED = 0

NB_EPISODES = 5
NB_STEPS = 200

OUTPUT_DIR = ROOT_DIR / "outputs" / "random_runs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

run_idx = len(os.listdir(OUTPUT_DIR))
current_output_dir = OUTPUT_DIR / f"run_{run_idx}"
current_output_dir.mkdir(parents=True, exist_ok=True)


def main() -> None:
    """Run the environment with a random agent."""
    # Create the environment
    env = ITHOREnv()
    env.reset(seed=SEED)
    callback = RecordVideoCallback(current_output_dir / "video.mp4")
    random_agent = RandomAgent(env, callback)

    # Run the agent
    for _ in range(NB_EPISODES):
        obs = random_agent.reset()
        random_agent.run_episode(obs, max_steps=NB_STEPS)

    # Close the agent
    random_agent.close()


if __name__ == "__main__":
    main()
