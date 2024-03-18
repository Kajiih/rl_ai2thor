"""Main module for AI2-THOR RL environment."""

from gymnasium.envs.registration import WrapperSpec, register

register(
    id="rl_ai2thor/ITHOREnv-v0.1",
    entry_point="rl_ai2thor.envs.ai2thor_envs:ITHOREnv",
    # TODO: Check if we add kwargs
    # TODO: Add additional_wrappers
)


NormalizeActionWrapper_spec = WrapperSpec(
    name="NormalizeActionWrapperSpec",
    entry_point="rl_ai2thor.envs.wrappers:NormalizeActionWrapper",
    kwargs={},
)

register(
    id="rl_ai2thor/ITHOREnv-v0.1_sb3_ready",
    entry_point="rl_ai2thor.envs.ai2thor_envs:ITHOREnv",
    additional_wrappers=(
        # WrapperSpec(
        #     name="NormalizeActionWrapper",
        #     entry_point="rl_ai2thor.envs.wrappers:NormalizeActionWrapper",
        #     kwargs={},
        # ),
        WrapperSpec(
            name="ChannelFirstObservationWrapper",
            entry_point="rl_ai2thor.envs.wrappers:ChannelFirstObservationWrapper",
            kwargs={},
        ),
    ),
)
