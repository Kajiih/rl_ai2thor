#!/bin/bash
if [ ! -e /proc/driver/nvidia/version ]; then
    echo "Error: Nvidia driver not found at /proc/driver/nvidia/version; Please ensure you have an Nvidia GPU device and appropriate drivers are installed."
    exit 1;
fi;

if  ! type "docker" 2> /dev/null > /dev/null ; then
    echo "Error: docker not found. Please install docker to complete the build. "
    exit 1
fi;

NVIDIA_VERSION=$(cat "/proc/driver/nvidia/version" | grep 'NVRM version:'| grep -oE "Kernel Module\s+[0-9.]+"| awk '{print $3}')
CUDA_VERSION=$(nvidia-smi | grep "CUDA Version" | awk '{print $9}')

if (id -nG | grep -qw "docker") || [ "$USER" == "root" ]; then
    echo "Building Docker container with CUDA Version: $CUDA_VERSION, NVIDIA Driver: $NVIDIA_VERSION"
    # docker build  --build-arg NVIDIA_VERSION="$NVIDIA_VERSION" --build-arg CUDA_VERSION="$CUDA_VERSION"  -t rlthor-docker:latest -f ./docker/Dockerfile .
    docker build  --build-arg NVIDIA_VERSION="$NVIDIA_VERSION" --build-arg CUDA_VERSION="$CUDA_VERSION"  -t rlthor-docker_test:latest -f ./docker/Dockerfile .
else
    echo "Error: Unable to run build.sh. Please use sudo to run build.sh or add $USER to the docker group."
    exit 1
fi

