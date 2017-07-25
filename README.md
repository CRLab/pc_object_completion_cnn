# pc_object_completion_cnn
ROS node for shape completion. Part of the point cloud processing framework https://github.com/CURG/pc_pipeline_launch, and the code base for the shape completion CNN to appear in IROS 2017 paper "Shape Completion Enabled Robotic Grasping".

```
@inproceedings{varley2017shape,
  title={Shape Completion Enabled Robotic Grasping},
  author={Varley, Jacob and DeChant, Chad and Richardson, Adam and Nair, Avinash and Ruales, Joaqu{\'\i}n and Allen, Peter},
  booktitle={Intelligent Robots and Systems (IROS), 2017 IEEE/RSJ International Conference on},
  year={2017},
  organization={IEEE}
}
```
## Dependencies
This has been tested on ubuntu 14.04 and ROS Indigo and ubuntu 16.04 and ROS Kinetic.  You will need to first setup Keras with a tensorflow backend.  See their documentation to do this. https://github.com/fchollet/keras/
```
non-ros:
git clone git@github.com:CURG/binvox-rw-py.git
git clone git@github.com:ShapeCompletion3D/python-pcl.git
git clone git@github.com:CURG/Curvox.git
git clone git@github.com:CURG/Mesh_Reconstruction.git

ros:
git clone git@github.com:CURG/pc_pipeline_msgs.git
```

<b>keras</b> (tensorflow backend): Deep learning library used to train the CNN, you need this to run a trained model as well.

<b>binvox_rw</b>: python utility library to read and write voxel grids to run length encoded binvox files

<b>python-pcl</b>: python utility library to read and write pointclouds to .pcd files

<b>Curvox</b>: python mesh/pointcloud utility library mostly for marshalling ros mesh and pointcloud messages into ply and pcd files

<b>Mesh_Reconstruction</b>: Code from IROS 2017 "Shape Completion Enabled Robotic Grasping" paper. This code takes the 40^3 voxel grid output from the CNN and combines it with the high resolution observed pointcloud directly captured from the depth sensor.  This way the visible portions of the completions are nicely detailed. 

## Setup
The weight file is too large to commit to github, so after cloning this repo, run the following to download a trained model from our server.
```
cd pc_object_completion_cnn/scripts/shape_completion_server/trained_models
./download_trained_model.sh
```

You will also need to modify set_env.sh to correctly set where to reach your ROS core. 

## Running
```
source set_env.sh
rosrun pc_object_completion_cnn mesh_completion_server.py
```
