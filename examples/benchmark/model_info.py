"""Model information for RL-THOR benchmark."""

from enum import StrEnum

from sb3_contrib import QRDQN
from stable_baselines3 import A2C, DQN, PPO


class ModelType(StrEnum):
    """SB3 compatible models."""

    PPO = "PPO"
    A2C = "A2C"
    DQN = "DQN"
    QRDQN = "QRDQN"
    RANDOM = "Random"


MODEL_CONFIG = {
    "verbose": 1,
    "progress_bar": True,
}


def get_model(model_name: ModelType) -> type[PPO] | type[A2C] | type[DQN] | type[QRDQN]:
    """Return the SB3 model class."""
    match model_name:
        case ModelType.PPO:
            return PPO
        case ModelType.A2C:
            return A2C
        case ModelType.DQN:
            return DQN
        case ModelType.QRDQN:
            return QRDQN
        case ModelType.RANDOM:
            raise ValueError("Random agent doesn't need a model.")
