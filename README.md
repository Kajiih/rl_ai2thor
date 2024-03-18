# RL THOR Benchmark

## How to use

## Install requirement

```bash
pip install -r requirements/base.txt

```

or

```bash

pip install -r requirements/base.txt

```

### Create custom environment mode

- Create a new mode config in `config/environment_modes/` folder and change `environment_mode` in `config/general.yaml` to the new mode name.

## Dev info

Virtual env: `rl_thor_benchmark`

## Possible features

- Add falling back on closest object to the agent if no object at the target coordinates
- Add falling back on closest object in the agent's field of view if no object at the target coordinates
- If no object is at the target object coordinates, add range around the coordinates to search for the object
- Add continuous/discrete dichotomy for some action groups only
- Add 3rd party/external camera
- Add enabling or disabling object properties from config (`openable`, `pickupable`, `receptacle`, ...)
- Add switching from on and off toggle actions to single toggle action from config

## TODOs

### Dev TODOs

-[`Started`] Add Gymnasium interface

-[`Started`] Create iTHOR environment class

- [ ] Refactor envs.tasks.tasks.py by using only item ids as dictionary key instead of sometimes the ids and sometimes the item itself
- [x] Refactor results and scores in envs.tasks.tasks.py to use 2 different dicts for relations and properties instead of 1 common dict to simplify types
- [ ] Add checking if actions are compatible with each other
  - [ ] Can't be 100% accurate, but we can do some checks (e.g.`hand_movements` implies `pickup`, ...)

  - [ ] Handle sliced items in tasks

    - [ ] In candidate objects

  - [ ] Add FlattenActionWrappers

    - [ ] Handle render modes

  - [`ToTest`] Implement target object through position
  - [ ] Add scene id handling at reset
  - [ ] Implement gray scale image mode
  - [`ToTest`] Implement variable openness
  - [`ToTest`] Handle default parameter value for actions

    - [`ToTest`] Handle changing default parameter value from config

  - [ ] Handle actions without arguments (in compute_compatible_task_args)
  - [ ] Create context manager to automatically close AI2-THOR window
  - [ ] Handle done actions
  - [x] Add reward
  - [ ] Find a way to add dense reward meaningfully
  - [x] Add reward for task completion
  - [ ] Handle episode termination (task success/failure)
  - [x] Handle truncation (timeout)
  - [ ] Handle depth frame and instance segmentation frame
  - [ ] Handle data saving from config
  - [ ] Handle step info better
  - [x] Check seed, randomization and reproducibility
  - [x] Handle reset at the end of the episode
  - [ ] Update docstring and documentation
  - [ ] Make `gridsize` and `moveMagnitude` parameters coherent (for `MoveAhead`-type actions)
  - [ ] Make the whole framework split-aware ('train', 'test' splits, ...)
  - [ ] Add some non-navigation mode? Where the agent can teleport while pointing at where it wants to go
  - [ ] Add parameters to relations and properties
  - [ ] Improve config (and environment mode config) handling
  - [ ] Add support for multitask with different information than text description (like index of task in a list (+index of target object types...?))
  - [x] Add Single task wrapper instead?
  - [x] Add normalize wrapper that normalizes observation (should not be needed) and actions: [-1, 1] for continuous
  - [x] Make wrapper to change obs from channel last to channel first
  - [ ] TODO: Improve environment registering (kwargs, wrappers..)
  - [ ] Make automatic version handling for the package and the environment register
  - [ ] Make Grayscale wrapper
  - [ ] Make some wrapper being the default behavior
  - [ ] Test combined wrappers
  - [ ] Change example scripts to jupyter notebooks
  - [ ] Add frame stacking
  - [ ] Add more SB3 algorithms
  - [ ] Add graph visualization with pygraphiz (does it need networkx?)

  - [ ] ManipulaTHOR env
  - [ ] RoboTHOR env

- [ ] Add RL algorithms

  - [ ] Support Stable Baselines3
  - [ ] Support RLlib..?
  - [ ] Support CleanRL

### Final TODOs

- [ ] Check the different config files
- [ ] Check and remove ipdb traces
- [ ] Check and remove asserts
- [ ] Check remaining TODOs in code
- [ ] Check and remove `noqa`, `type: ignore` and `sourcery`
- [ ] Check and remove unused code
- [ ] Check and remove unused dependencies
- [ ] Check and remove unused files
- [ ] Test the framework with clean install
- [ ] Double check docstrings and comments
- [ ] Add documentation
- [ ] Add Pre-commit hooks
- [ ] Write changelog (<https://keepachangelog.com/en/1.0.0/>)
- [ ] Double check pyproject.toml file, especially dependencies

## Interesting Github Issues

- <https://github.com/allenai/ai2thor/issues/993>
- <https://github.com/allenai/ai2thor/issues/1144>
- <https://github.com/allenai/ai2thor/issues/21>
- <https://github.com/allenai/ai2thor/issues/851>
- <https://github.com/allenai/ai2thor-docker/issues/3>
- <https://github.com/allenai/ai2thor-docker/issues/2>
- <https://github.com/allenai/ai2thor-docker/issues/14>
- <https://github.com/allenai/ai2thor-docker/issues/9>
