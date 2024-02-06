from gymnasium.envs.registration import register

register(
    id="gym_ai2thor/iTHOR-v0",
    entry_point="rl_ai2thor.envs.ai2thor_envs:iTHOREnv",
    max_episode_steps=300,  # TODO: Check if we keep this
)

# %%
