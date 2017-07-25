# pc_object_completion_cnn
ROS node for shape completion. Part of the point cloud processing framework https://github.com/CURG/pc_pipeline_launch

## Dependencies
This has been tested on ubuntu 14.04 and ROS Indigo and ubuntu 16.04 and ROS Kinetic
```
non-ros:
git clone https://github.com/fchollet/keras/ 
git clone https://github.com/CURG/binvox-rw-py
git clone https://github.com/ShapeCompletion3D/python-pcl
git clone https://github.com/CURG/Curvox
git clone https://github.com/CURG/Mesh_Reconstruction

ros:
pc_pipeline_msgs: https://github.com/CURG/pc_pipeline_msgs
```

keras (tensorflow backend): Deep learning library used to train the CNN, you need this to run a trained model as well.
binvox_rw: python utility library to read and write voxel grids to run length encoded binvox files
python-pcl: python utility library to read and write pointclouds to .pcd files
curvox: python mesh/pointcloud utility library mostly for marshalling ros mesh and pointcloud messages into ply and pcd files
Mesh_Reconstruction: Code from IROS 2017 "Shape Completion Enabled Robotic Grasping" paper. This code takes the 40^3 voxel grid output from the CNN and combines it with the high resolution observed pointcloud directly captured from the depth sensor.  This way the visible portions of the completions are nicely detailed. 

## Setup
The weight file is too large to commit to github, so after cloning this repo, run the following to download a trained model from our server.
```
cd pc_object_completion_cnn/scripts/shape_completion_server/trained_models
./download_trained_model.sh
```
