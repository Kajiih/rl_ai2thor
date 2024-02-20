"""Run a random agent in the AI2Thor RL environment."""

# %%
import os

import imageio

from rl_ai2thor.envs.ai2thor_envs import ITHOREnv
from rl_ai2thor.utils.general_utils import ROOT_DIR

NB_STEPS = 1000

OUTPUT_DIR = ROOT_DIR / "outputs" / "random_agent"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

run_idx = len(os.listdir(OUTPUT_DIR))
current_output_dir = OUTPUT_DIR / f"run_{run_idx}"
current_output_dir.mkdir(parents=True, exist_ok=True)

video_writer = imageio.get_writer(current_output_dir / "video.mp4", fps=10)


def main() -> None:
    """Run the environment with a random agent."""
    # Create the environment
    env = ITHOREnv()
    env.reset()
    # Run the agent
    terminated = False
    obs, info = env.reset()
    for _i in range(NB_STEPS):
        # Select a random action
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"Action: {action}, Reward: {reward}, Terminated: {terminated}")

        # im = Image.fromarray(obs)
        # im.save(current_output_dir / f"frame_{i}.png")
        video_writer.append_data(obs)

        # TODO: Add saving frames and other data
    env.close()
    video_writer.close()


if __name__ == "__main__":
    main()
