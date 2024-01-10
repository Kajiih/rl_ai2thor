# RL THOR Benchmark

## How to use

### Create custom emvironment mode

- Create a new mode config in `config/enviroment_modes/` folder and change `environment_mode` in `config/general.yaml` to the new mode name.

## Dev info

Virutal env: `rl_thor_benchmark`

## Possible features

- Add falling back on closest object to the agent if no object at the target coordinates
- Add falling back on closest object in the agent's field of view if no object at the target coordinates
- If no object is at the target object coordinates, add range around the coordinates to search for the object
- Add continuous/discrete dichotomy for some action groups only
- Add 3rd pary/external camera
- Add enabling or disabling object properties from config (`openable`, `pickupable`, `receptacle`, ...)
- Add switching from on and off toggle actions to single toggle action from config

## TODOs

### Dev TODOs

- [`Started`] Add Gymnasium interface

  - [`Started`] Create iTHOR environment class

    - [] Add FlattenActionWrapper
      - [] Handle render modes
    - [`ToTest`] Implement target object through position
    - [] Add scene id handling at reset
    - [] Implement grayscale image mode
    - [`ToTest`] Implement variable openness
    - [`ToTest`] Handle default parameter value for actions
      - [`ToTest`] Handle changing default parameter value from config
    - [] Handle done actions
    - [] Add reward
    - [] Handle episode termination (task success/failure)
    - [`ToTest`] Handle truncation (timeout)
    - [] Handle depth frame and instance segmentation frame
    - [] Handle data saving from config
    - [] Handle step info better
    - [] Check seed, randomization and reproducibility
    - [] Handle reset at the end of the episode
    - [] Update docstrings and documentation
    - [] Make gridsize and moveMagnitude paramters coherent (for `MoveAhead`-type actions)
    - [] Make the whole framework split-aware ('train', 'test' splits, ...)

  - [] ManipulaTHOR env
  - [] RoboTHOR env

- [] Add RL algorithms
  - [] Support Stable Baselines3
  - [] Support RLlib..?
  - [] Support CleanRL

### Final TODOs

- [] Check the different config files
- [] Check and remove ipdb traces
- [] Check and remove asserts
- [] Check remaining TODOs in code
- [] Check and remove unused code
- [] Check and remove unused dependencies
- [] Check and remove unused files
- [] Test the framework with clean install
- [] Double docstrings and comments
- [] Add documentation

## Interesting Github Issues

- https://github.com/allenai/ai2thor/issues/993
- https://github.com/allenai/ai2thor/issues/1144
- https://github.com/allenai/ai2thor/issues/21
- https://github.com/allenai/ai2thor/issues/851
- https://github.com/allenai/ai2thor-docker/issues/3
- https://github.com/allenai/ai2thor-docker/issues/2
- https://github.com/allenai/ai2thor-docker/issues/14
- https://github.com/allenai/ai2thor-docker/issues/9
