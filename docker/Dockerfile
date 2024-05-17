ARG CUDA_VERSION=12.2.2

FROM nvidia/cuda:$CUDA_VERSION-devel-ubuntu22.04
FROM python:3.12
ARG NVIDIA_VERSION

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get -y install python3-pip libxrender1 libsm6 xserver-xorg-core xorg python3-venv vim pciutils wget git kmod vim git

ENV APP_HOME /app
WORKDIR $APP_HOME
COPY docker/scripts/install_nvidia.sh /app/
RUN pip3 install --upgrade pip

COPY requirements ./requirements
RUN pip3 install -r requirements/dev.txt

RUN NVIDIA_VERSION=$NVIDIA_VERSION /app/install_nvidia.sh

# ADD "https://www.random.org/cgi-bin/randbyte?nbytes=10&format=h" skipcache
COPY docker/ai2thor_docker /app/ai2thor_docker
COPY src /app/src
COPY config /app/config
COPY examples /app/examples

# Set PYTHONPATH to include the rl_thor package directory
ENV PYTHONPATH="${PYTHONPATH}:/app/src"

# Temp
COPY docker/example_agent.py ./


CMD export DISPLAY=:0.0 && ai2thor-xorg start && /bin/bash
