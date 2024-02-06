"""Run a random agent in the AI2Thor RL environment."""

# %%
import gymnasium as gym

from ai2thor_envs import ITHOREnv

NB_STEPS = 100


def main():
    # Create the environment
    env = ITHOREnv()
    env.reset()
    # Run the agent
    terminated = False
    obs, info = env.reset()
    for _ in range(NB_STEPS):

        # Select a random action
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"Action: {action}, Reward: {reward}, Terminated: {terminated}")

        # TODO: Add saving frames and other data
    env.close()


if __name__ == "__main__":
    main()

# %%
