"""Test the loading of the environment."""

import gymnasium as gym

from rl_thor.envs import ITHOREnv


def main() -> None:
    """Test the loading of the environment."""
    print("Creating the environment from the ITHOREnv class")
    env = ITHOREnv()
    print("Success")

    print("Making environment from registered id")
    env = gym.make("rl_thor/ITHOREnv-v0.1")
    print("Success")


if __name__ == "__main__":
    main()
