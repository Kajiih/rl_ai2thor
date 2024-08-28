#!/bin/bash
CUSTOM_COMMAND="$1"

DIR=$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )
cd "$DIR"/../ || exit


# Inference on train split
X11_PARAMS=""
if [[ -e /tmp/.X11-unix && ! -z ${DISPLAY+x} ]]; then
    echo "Using local X11 server"
    X11_PARAMS="-e DISPLAY=$DISPLAY  -v /tmp/.X11-unix:/tmp/.X11-unix:rw"
    xhost +local:root
fi;


mkdir -p ai2thor_simulator_resources
mkdir -p ../runs
mkdir -p ../checkpoints
mkdir -p ../wandb

docker run --privileged $X11_PARAMS -it \
--mount type=bind,source="$(pwd)"/ai2thor_simulator_resources/,target=/root/.ai2thor/ \
--mount type=bind,source="$(pwd)"/../runs/,target=/app/runs/ \
--mount type=bind,source="$(pwd)"/../checkpoints/,target=/app/checkpoints/ \
--mount type=bind,source="$(pwd)"/../wandb/,target=/app/wandb/ \
--mount type=bind,source="$(pwd)"/../examples/benchmark/script/,target=/app/scripts/ \
rlthor-docker:latest /bin/bash -c "export DISPLAY=:0.0 && ai2thor-xorg start && $CUSTOM_COMMAND"


if [[ -e /tmp/.X11-unix && ! -z ${DISPLAY+x} ]]; then
    xhost -local:root
fi;

