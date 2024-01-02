"""
Gymnasium interface for ai2thor environment

Based on the code from cups-rl: https://github.com/TheMTank/cups-rl (MIT License)
# TODO: Check if we keep this
"""

import ai2thor.controller
import gymnasium as gym


ithor_actions = ["MoveAhead",]



class IThorEnv(gym.Env):
    """
    Wrapper base class for iTHOR enviroment
    """

    def 