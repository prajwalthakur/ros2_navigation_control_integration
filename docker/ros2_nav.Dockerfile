FROM  nvidia/cuda:12.6.0-cudnn-devel-ubuntu22.04

RUN apt-get update && apt-get upgrade -y

# https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/docker-specialized.html

ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=all

RUN apt-get install --no-install-recommends -y \
    software-properties-common \
    vim \
    python3-pip\
    tmux \
    git 

# Added updated mesa drivers for integration with cpu - https://github.com/ros2/rviz/issues/948#issuecomment-1428979499
RUN add-apt-repository ppa:kisak/kisak-mesa && \
    apt-get update && apt-get upgrade -y &&\
    apt-get install libxcb-cursor0 -y && \
    apt-get install ffmpeg python3-opengl -y

RUN pip3 install matplotlib PyQt5 dill pandas pyqtgraph transforms3d

RUN add-apt-repository universe

RUN DEBIAN_FRONTEND=noninteractive TZ=Etc/UTC apt-get -y install tzdata

RUN apt-get install -y curl && \
     curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o /usr/share/keyrings/ros-archive-keyring.gpg && \
     echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(. /etc/os-release && echo $UBUNTU_CODENAME) main" | tee /etc/apt/sources.list.d/ros2.list > /dev/null && \
     apt-get update && apt-get install -y ros-humble-ros-base


RUN apt-get install --no-install-recommends -y ros-humble-rviz2

ENV ROS_DISTRO=humble

# # Cyclone DDS
# RUN apt-get update --fix-missing 
RUN apt-get install --no-install-recommends -y \
    ros-$ROS_DISTRO-cyclonedds \
    ros-$ROS_DISTRO-rmw-cyclonedds-cpp



# Use cyclone DDS by default
ENV RMW_IMPLEMENTATION=rmw_cyclonedds_cpp

# install turtlebot3 sim files and gazebo
# ros-humble-navigation2 ros-humble-nav2-bringup 
# Install required tools
RUN apt-get update && apt-get install -y curl gnupg lsb-release

# Add GPG key and repository
RUN curl -fsSL https://packages.osrfoundation.org/gazebo.gpg \
    -o /usr/share/keyrings/pkgs-osrf-archive-keyring.gpg && \
    echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/pkgs-osrf-archive-keyring.gpg] \
    http://packages.osrfoundation.org/gazebo/ubuntu-stable $(lsb_release -cs) main" \
    > /etc/apt/sources.list.d/gazebo-stable.list

# 1. Gazebo Classic simulator **and** the CMake/headers that C++ projects need
RUN apt-get update && apt install gazebo libgazebo-dev -y      

# 2. ROS glue layers (plugins, launch files, ROS–Gazebo factory)
RUN apt-get update && apt install ros-humble-gazebo-ros-pkgs  -y 
RUN apt-get update && apt install ros-humble-gazebo-ros2-control -y 
RUN apt-get update && install ros-humble-rqt-graph -y
RUN apt-get update && apt install \
ros-humble-ros2-control -y \
ros-humble-gazebo-ros2-control -y \
ros-humble-ros2-controllers -y \
ros-humble-ros2_control_cmake -y \
ros-humble-ros-gz -y \
ros-humble-gz-ros2-control -y 


# Install Ignition Fortress
RUN apt-get update && apt-get install -y ignition-fortress


# RUN apt install ros-humble-turtlebot3* -y
# Source by default
RUN echo "source /opt/ros/$ROS_DISTRO/setup.bash" >> /root/.bashrc
RUN echo "source /root/workspace/install/setup.bash" >> /root/.bashrc
RUN echo "export TURTLEBOT3_MODEL=burger">>/root/.bashrc
RUN pip3 install -U colcon-common-extensions \
    && apt-get install -y build-essential python3-rosdep

RUN pip3 install --no-cache-dir Cython


#install jax
RUN pip3 install  "jax[cuda12]" 

#if want to install jax-cpu
#RUN pip3 install "jax[cpu]" -f https://storage.googleapis.com/jax-releases/jax_releases.html

RUN git clone https://github.com/prajwalthakur/ghalton.git && cd ghalton && pip install -e.

# Copy workspace files
ENV WORKSPACE_PATH=/root/workspace
COPY workspace/ $WORKSPACE_PATH/src/


# Set shell to bash
SHELL ["/bin/bash", "-c"]

# Final cleanup
RUN apt-get clean && rm -rf /var/lib/apt/lists/*

# Set default shell to bash
CMD ["/bin/bash"]
