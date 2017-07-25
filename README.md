# pc_object_completion_cnn
ROS node for shape completion. Part of the point cloud processing framework https://github.com/CURG/pc_pipeline_launch

## Dependencies
This has been tested on ubuntu 14.04 and ROS Indigo and ubuntu 16.04 and ROS Kinetic
```
non-ros:
keras (tensorflow backend): https://github.com/fchollet/keras/ 
binvox_rw: https://github.com/CURG/binvox-rw-py
python-pcl: https://github.com/ShapeCompletion3D/python-pcl
curvox: https://github.com/CURG/Curvox

ros:
pc_pipeline_msgs: https://github.com/CURG/pc_pipeline_msgs
```

## Setup
The weight file is too large to commit to github, so after cloning this repo, run the following to download a trained model from our server.
```
cd pc_object_completion_cnn/scripts/shape_completion_server/trained_models
./download_trained_model.sh
```
