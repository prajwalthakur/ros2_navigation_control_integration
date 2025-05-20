#!/bin/bash

# Get base.sh funcs
source "$(dirname "$0")/base.sh"

stop_docker

run_docker ros2_nav:latest "/root/app.sh"