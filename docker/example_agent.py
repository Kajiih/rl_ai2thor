"""Example agent in RL-THOR docker."""

from pprint import pprint

import ai2thor.controller
from ai2thor_docker.x_server import startx

if __name__ == "__main__":
    startx()
    controller = ai2thor.controller.Controller(scene="FloorPlan1")
    controller.step(action="MoveAhead")
    event = controller.step(action="PickupObject", objectId="Mug|-01.76|+00.90|-00.62", forceAction=True)
    event = controller.step(action="RotateRight")
    pprint(event.metadata["agent"])
    print("Execution successful!")
