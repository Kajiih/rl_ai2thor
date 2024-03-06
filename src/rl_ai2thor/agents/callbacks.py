"""Callbacks for RL AI2THOR agents."""

from pathlib import Path
from typing import Any

import imageio


# %% === Callbacks ===
class BaseCallback:
    """Base class for callbacks."""

    def __init__(self, verbose: int = 1) -> None:
        self.verbose = verbose

    def on_step(
        self,
        obs: Any = None,
        reward: float = 0,
        terminated: bool = False,
        truncated: bool = False,
        info: dict[str, Any] | None = None,
    ) -> None:
        """Triggered after each step."""

    def on_close(self) -> None:
        """Triggered when the agent is closed."""

    def on_reset(self) -> None:
        """Triggered when the agent is reset."""


class RecordVideoCallback(BaseCallback):
    """Callback to record a video of the environment."""

    def __init__(self, path_to_write: str | Path, frame_rate: int = 30) -> None:
        """Initialize the callback."""
        super().__init__()
        self.video_writer = imageio.get_writer(path_to_write, fps=frame_rate)
        Path(path_to_write).parent.mkdir(parents=True, exist_ok=True)

    def on_step(
        self,
        obs: Any,
        reward: float = 0,  # noqa: ARG002
        terminated: bool = False,  # noqa: ARG002
        truncated: bool = False,  # noqa: ARG002
        info: dict[str, Any] | None = None,  # noqa: ARG002
    ) -> None:
        """Record the observation."""
        self.video_writer.append_data(obs)

    def on_close(self) -> None:
        """Close the video writer."""
        self.video_writer.close()
