# AI2-THOR RL Benchmark

AI2-THOR RL is a lightweight and widely customizable reinforcement learning environment based on [AI2-THOR](https://ai2thor.allenai.org/) environment simulation. It provides a rich set of predefined tasks and allows users to create custom tasks and settings for the environment. The environment supports both continuous and discrete action spaces, and provides various observation modalities such as 1st or 3rd person RGB images, depth maps, and instance segmentation masks **[not supported yet, check if we keep this]**. It also supports multi-task learning and integrates seamlessly with popular RL algorithms implementations like [Stable Baselines3](https://github.com/DLR-RM/stable-baselines3).

AI2-THOR is a photorealistic interactive 3D environment for training AI agents that understand the world in the same way humans do. It is designed to be a general-purpose environment for training AI agents, and it is widely used by the research community. This project aims at providing a simple and efficient way to use AI2-THOR for reinforcement learning research.

**[Continue with more details]** -
**[Add gif of agent training]** -

## Contents <!-- omit from toc -->

- [Installation](#installation)
- [Getting Started](#getting-started)
- [Running Headless](#running-headless)
- [Configuring the environment](#configuring-the-environment)
- [Creating new tasks](#creating-new-tasks)
- [Reproducing benchmark results](#reproducing-benchmark-results)
- [Citation](#citation)
- [License](#license)

## Installation

1. **Create virtual environment**
    We recommend you use a [conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/) virtual environment:

    ```bash
      #  We require python>=3.12 
      conda create -n rl_ai2thor python=3.12
      conda activate rl_ai2thor
    ```

2. **Install AI2-THOR RL and its dependencies**
    To install and customize the environment locally:

    ```bash
    git clone https://github.com/Kajiih/rl_ai2thor.git
    pip install -r requirements/dev.txt
    ```

    **[!! Not supported yet !!]**
    Alternatively, if you only want to use predefined settings and task, you can install the PyPI package:

    ```bash
    pip install rl-ai2thor
    ```

## Getting Started

AI2-THOR RL uses [Gymnasium](https://github.com/Farama-Foundation/Gymnasium) API, so you can use it as simply as any other Gym/Gymnasium environment.

This short script runs an ITHOR environment with the basic configuration and random actions:

```python
import gymnasium as gym

env = gym.make("rl_ai2thor/ITHOREnv-v0.1")
obs, info = env.reset()

terminated, truncated = False, False
while not terminated or truncated:
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
env.close()
```

Note that the first time that you instantiate the environment, AI2-THOR will download the simulator resources.

Follow Jupyter Notebooks in `examples` for more complex examples and use cases, like using AI2-THOR RL with Stable Baselines3 or recording videos of the agent's view.

To go further, we recommend you to get familiar with the [concepts of the ITHOR simulation environment](https://ai2thor.allenai.org/ithor/documentation/concepts) and [our documentation](https://github.com/Kajiih/rl_ai2thor) **[create actual documentation]** to understand how tasks are defined.

## Running Headless

Not available yet

## Configuring the environment

The general environment configuration, like the maximum number of steps by episode or the parameters of the AI2-THOR controller can be found and edited in `config/general.yaml`.

To change which actions can the agent use or the set of tasks and scenes it is trained on, you can create a new environment mode config file in `config/environment_modes/` or use one of the preset and change the `environment_mode` value in the general config.

**Example:**
After creating your custom environment mode config file in `config/environment_modes/custom_env_mode.yaml`, you will need to write:

```yaml
environment_mode: custom_env_mode
```

You can also set your own config folder containing `general.yaml` and `environment_modes` when instantiating the environment with the `config_folder_path` argument. It is especially useful when you use the PyPI package.
Additionally, you can override only certain values of the general or environment mode config with the `override_config` argument.

**Example:**
If you only want to change the maximum number of steps per episode to 500 and exclude all `Kitchen` scenes from all task, you can write:

```python
override_config = {
    "max_episode_steps": 500,
    "globally_excluded_scenes": ["Kitchen"],
}
env = gym.make(
    "rl_ai2thor/ITHOREnv-v0.1",
    config_folder_path="config",  # Default value
    override_config=override_config,
)
```

For more details about the configuration, see the [dedicated part of the documentation](https://github.com/Kajiih/rl_ai2thor) **[create actual documentation]**.

## Creating new tasks

In AI2-THOR RL, we use a specific task description format called [Graph Task](https://github.com/Kajiih/rl_ai2thor) **[link to documentation]**.

Thanks to graph tasks, defining new tasks is as simple as creating a python dictionary describing the task items, their properties and their relations:

```python
task_description_dict = {
    "hot_apple": {
        "properties": {"objectType": "Apple", "temperature": "Hot"},
    },
    "plate_receptacle": {
        "properties": {"objectType": "Plate"},
        "relations": {"hot_apple": ["contains"]},
    },
}
```

This code lets you define a new task consisting of putting a hot apple in a plate. `hot_apple` and `plate_receptacle` are identifiers of the items used to defined relations and each property and relation can be found [here](.) **[link to documentation]**. This is enough to automatically create the reward function associated to the graph task.

For more details about how to define new tasks, item properties or relations, see the [dedicated part of the documentation](.) **[create actual documentation]**.

## Reproducing benchmark results

Not available yet

## Citation

Not available yet

## License

| Component            | License                                                                  |
| -------------------- | -------------------------------------------------------------------------|
| Codebase (this repo) | [MIT License](LICENSE)                                                   |
| AI2-THOR             | [Apache License Version 2.0](https://www.apache.org/licenses/LICENSE-2.0)|
