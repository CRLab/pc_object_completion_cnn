#!/bin/bash

# set ROS env variables
export ROS_MASTER_URI=http://skye.local:11311
echo ROS_MASTER_URI $ROS_MASTER_URI

export ROS_HOSTNAME=$(hostname -f).local
echo ROS_HOSTNAME $ROS_HOSTNAME
