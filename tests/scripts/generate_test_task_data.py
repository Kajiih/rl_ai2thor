"""Generate data for the tasks tests."""

# %% === Setup the environment for the test tasks ===
import pickle as pkl  # noqa: S403
from pathlib import Path

import ai2thor.controller

data_dir = Path("../data")

controller = ai2thor.controller.Controller()


# %% === Test Pickup task with a mug ===
test_data_dir = data_dir / "test_pickup_mug"
test_data_dir.mkdir(parents=True, exist_ok=True)

event_list_path = test_data_dir / "event_list.pkl"
advancement_list_path = test_data_dir / "advancement_list.pkl"
terminated_list_path = test_data_dir / "terminated_list.pkl"
event_list = []
advancement_list = []
terminated_list = []

event = controller.reset("FloorPlan1")
event_list.append(event)
advancement_list.append(1)
terminated_list.append(False)

event = controller.step(action="PickupObject", objectId="Apple|-00.47|+01.15|+00.48", forceAction=True)
event_list.append(event)
advancement_list.append(1)
terminated_list.append(False)

event = controller.step(action="DropHandObject", forceAction=True)

event = controller.step(action="PickupObject", objectId="Mug|-01.76|+00.90|-00.62", forceAction=True)
event_list.append(event)
advancement_list.append(2)
terminated_list.append(True)


event = controller.step(action="DropHandObject", forceAction=True)
event_list.append(event)
advancement_list.append(1)
terminated_list.append(False)

with event_list_path.open("wb") as f, advancement_list_path.open("wb") as g, terminated_list_path.open("wb") as h:
    pkl.dump(event_list, f)
    pkl.dump(advancement_list, g)
    pkl.dump(terminated_list, h)

# %% === Test Open task with a Fridge ===
test_data_dir = data_dir / "test_open_fridge"
test_data_dir.mkdir(parents=True, exist_ok=True)

event_list_path = test_data_dir / "event_list.pkl"
advancement_list_path = test_data_dir / "advancement_list.pkl"
terminated_list_path = test_data_dir / "terminated_list.pkl"
event_list = []
advancement_list = []
terminated_list = []

event = controller.reset("FloorPlan1")
event_list.append(event)
advancement_list.append(1)
terminated_list.append(False)

event = controller.step(action="OpenObject", objectId="Drawer|-01.56|+00.66|-00.20", forceAction=True)
event_list.append(event)
advancement_list.append(1)
terminated_list.append(False)

event = controller.step(action="CloseObject", objectId="Drawer|-01.56|+00.66|-00.20", forceAction=True)

event = controller.step(action="OpenObject", objectId="Fridge|-02.10|+00.00|+01.07", forceAction=True)
event_list.append(event)
advancement_list.append(2)
terminated_list.append(True)


event = controller.step(action="CloseObject", objectId="Fridge|-02.10|+00.00|+01.07", forceAction=True)
event_list.append(event)
advancement_list.append(1)
terminated_list.append(False)

with event_list_path.open("wb") as f, advancement_list_path.open("wb") as g, terminated_list_path.open("wb") as h:
    pkl.dump(event_list, f)
    pkl.dump(advancement_list, g)
    pkl.dump(terminated_list, h)


# %% === Test place cooled in with a apple and counter top ===

test_data_dir = data_dir / "test_place_cooled_in_apple_counter_top"
test_data_dir.mkdir(parents=True, exist_ok=True)

event_list_path = test_data_dir / "event_list.pkl"
advancement_list_path = test_data_dir / "advancement_list.pkl"
terminated_list_path = test_data_dir / "terminated_list.pkl"
event_list = []
advancement_list = []
terminated_list = []

event = controller.reset("FloorPlan1")
event_list.append(event)
advancement_list.append(4)
terminated_list.append(False)

event = controller.step(action="OpenObject", objectId="Fridge|-02.10|+00.00|+01.07", forceAction=True)
event_list.append(event)
advancement_list.append(4)
terminated_list.append(False)

event = controller.step(action="PickupObject", objectId="Apple|-00.47|+01.15|+00.48", forceAction=True)
event_list.append(event)
advancement_list.append(2)
terminated_list.append(False)

event = controller.step(
    action="PutObject",
    objectId="Fridge|-02.10|+00.00|+01.07",
    forceAction=True,
)
event_list.append(event)
advancement_list.append(3)
terminated_list.append(False)

event = controller.step(action="PickupObject", objectId="Apple|-00.47|+01.15|+00.48", forceAction=True)
event_list.append(event)
advancement_list.append(3)
terminated_list.append(False)

event = controller.step(
    action="PutObject",
    objectId="CounterTop|+00.69|+00.95|-02.48",
    forceAction=True,
)
event_list.append(event)
advancement_list.append(5)
terminated_list.append(True)

event = controller.step(action="CloseObject", objectId="Fridge|-02.10|+00.00|+01.07", forceAction=True)
event_list.append(event)
advancement_list.append(5)
terminated_list.append(True)

event = controller.step(action="PickupObject", objectId="Apple|-00.47|+01.15|+00.48", forceAction=True)
event_list.append(event)
advancement_list.append(3)
terminated_list.append(False)

with event_list_path.open("wb") as f, advancement_list_path.open("wb") as g, terminated_list_path.open("wb") as h:
    pkl.dump(event_list, f)
    pkl.dump(advancement_list, g)
    pkl.dump(terminated_list, h)


# %% === Stop the controller ===
controller.stop()

# %%
