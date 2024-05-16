from ai2thor_docker.x_server import startx
import ai2thor.controller
import os
import time
from pprint import pprint


if __name__ == "__main__":
    startx()
    controller = ai2thor.controller.Controller(scene="FloorPlan1")
    controller.step(action="MoveAhead")
    event = controller.step(action="PickupObject", objectId="Mug|-01.76|+00.90|-00.62", forceAction=True)
    event = controller.step(action="RotateRight")
    print("Example finished!")
    pprint(event.metadata["agent"])
