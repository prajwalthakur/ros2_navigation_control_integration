#!/bin/bash
PROJECT_ROOT="$PWD"
# absolute_path=/home/prajwal/projects/ros2_nav_for_ROS2_Simulation/
run_docker() {
    # -it is for interactive, tty
    # --privileged for accessing /dev contents
    # --net=host to share the same network as host machine. TL;DR same IP.
    xhost +local:root # giving display privilages
    docker run -it --privileged --net=host \
    --name ros2_nav \
    --env="DISPLAY" \
    --env="QT_X11_NO_MITSHM=1" \
    -v ${PROJECT_ROOT}/scripts/deploy/app.sh:/root/app.sh \
    $@
}



stop_docker() {
    docker stop ros2_nav && docker rm ros2_nav
}
