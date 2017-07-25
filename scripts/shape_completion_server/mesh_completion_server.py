#!/usr/bin/env python

import os
import time
import shutil
import importlib
import numpy as np
import subprocess
import argparse
import rospkg

import pcl
import binvox_rw
import plyfile
import mcubes
import plyfile
import geometry_msgs.msg
import shape_msgs.msg

import tf
import actionlib
import geometry_msgs
import shape_msgs
import shape_completion_msgs.msg
import rospy
import PyKDL
import tf_conversions
import shape_completion_server.srv

import curvox.utils, curvox.mesh_conversions

import tempfile
from sensor_msgs import point_cloud2


class MeshCompletionServer(object):

    def __init__(self, cnns):

        rospy.loginfo("Starting Completion Server")

        self.patch_size = 40
        self.cnns = cnns
        self.cnn_python_module = cnns["depth"]["cnn_python_module"]
        self.weights_filepath = cnns["depth"]["weights_filepath"]

        self.post_process_executable = "mesh_reconstruction"
        
        self._feedback = shape_completion_msgs.msg.CompleteMeshFeedback()
        self._result = shape_completion_msgs.msg.CompleteMeshResult()

        self._as = actionlib.SimpleActionServer("/object_completion", shape_completion_msgs.msg.CompleteMeshAction,
            execute_cb=self.completion_cb, auto_start=False)

        self._switch_cnn_type_srv = rospy.Service(
            '/shape_completion_server/set_cnn_type', 
            shape_completion_server.srv.SetCNNType,
            self.set_cnn_type
        )

        self._as.start()

        rospy.loginfo("Started Completion Server")

    def complete_voxel_grid(self, batch_x_B012C):

       	py_module = importlib.import_module(self.cnn_python_module)
        self.model= py_module.get_model()
        self.model.load_weights(self.weights_filepath)

        # The new version of keras takes the data as B012C
        # NOT BZCXY so we do not need to transpose it.
        batch_x = batch_x_B012C

        # run the batch through the network and get the completion
        # pred is actually flat, because we do not have a reshape as the final
        # layer of the net, so pred's shape is something like
        # (batch_size, 40*40*40)

        pred = self.model.predict(batch_x)
        
        # The new version of keras takes the data as B012C
        # NOT BZCXY so we do not need to transpose it.
        pred_as_b012c = pred.reshape(1, self.patch_size, self.patch_size, self.patch_size, 1) 
        completed_region = pred_as_b012c[0, :, :, :, 0]
        
        return completed_region

    def set_cnn_type(self, request):
        cnn_name = request.cnn_name
        if cnn_name not in self.cnns:
            return shape_completion_server.srv.SetCNNTypeResponse(success=False)

        self.cnn_python_module = cnns[cnn_name]["cnn_python_module"]
        self.weights_filepath = cnns[cnn_name]["weights_filepath"]

        return shape_completion_server.srv.SetCNNTypeResponse(success=True)

    def completion_cb(self, goal):
        rospy.loginfo('Received Completion Goal')

        self._feedback = shape_completion_msgs.msg.CompleteMeshFeedback()
        self._result = shape_completion_msgs.msg.CompleteMeshResult()
        temp_pcd_handle, temp_pcd_filepath = tempfile.mkstemp(suffix=".pcd")
    
        pc = point_cloud2.read_points(goal.partial_cloud)
        partial_pc_np = np.asarray(list(pc))
        pcd = pcl.PointCloud(np.array(partial_pc_np[:, 0:3],np.float32))
        pcl.save(pcd, temp_pcd_filepath)

        batch_x = np.zeros((1, self.patch_size, self.patch_size, self.patch_size, 1), dtype=np.float32)
        batch_x[0, :, :, :, :], voxel_resolution, offset = curvox.utils.build_test_from_pc_scaled(partial_pc_np[:, 0:3], self.patch_size)        

        batch_x = batch_x.transpose(0,2,1,3,4)
        batch_x_new = np.zeros_like(batch_x)
        for i in range(40):
            batch_x_new[0, i, :, :, 0] = batch_x[0, 40-i-1, :, :, 0]

        batch_x = batch_x_new

        # output is the completed voxel grid,
        # it is all floats between 0,1 as last layer is softmax
        # think of this as probability of occupancy per voxel
        # output.shape = (X,Y,Z)
        output = self.complete_voxel_grid(batch_x)

        output_new = np.zeros_like(output)
        for i in range(40):
            output_new[i, :, :] = output[40-i-1, :, :]

        output = output_new

        output = output.transpose(1, 0, 2)

        # mask the output, so above 0.5 is occupied
        # below 0.5 is empty space. 
        output_vox = np.array(output)>0.5

        # Save the binary voxel grid as an occupancy map
        # in a binvox file
        vox = binvox_rw.Voxels(output_vox,
                           (self.patch_size, self.patch_size, self.patch_size),
                            (offset[0], offset[1], offset[2]),
                           voxel_resolution* self.patch_size,
                           "xyz")

        # Now we save the binvox file so that it can be passed to the
        # post processing along with the partial.pcd 
        _, temp_binvox_filepath = tempfile.mkstemp(suffix="output.binvox")                                                                                                                      
        binvox_rw.write(vox, open(temp_binvox_filepath, 'w'))

        # mask the output, so above 0.5 is occupied
        # below 0.5 is empty space. 
        input_vox_arr = np.array(batch_x[0, :, :, :, 0])>0.5

        # Save the binary voxel grid as an occupancy map
        # in a binvox file
        input_vox = binvox_rw.Voxels(input_vox_arr,
                           (self.patch_size, self.patch_size, self.patch_size),
                            (offset[0], offset[1], offset[2]),
                           voxel_resolution* self.patch_size,
                           "xyz")

        # Now we save the binvox file so that it can be passed to the
        # post processing along with the partial.pcd 
        _, temp_input_binvox_file = tempfile.mkstemp(suffix="input.binvox")                                                                                                                      
        binvox_rw.write(input_vox, open(temp_input_binvox_file, 'w'))

        # This is the file that the post-processed mesh will be saved it. 
        _, temp_completion_filepath = tempfile.mkstemp(suffix=".ply")

        # This command will look something like
        # mesh_reconstruction tmp/completion.binvox tmp/partial.pcd tmp/post_processed.ply
        cmd_str = self.post_process_executable + " " + temp_binvox_filepath + " " + temp_pcd_filepath \
                  + " " + temp_completion_filepath + " --cuda"

        subprocess.call(cmd_str.split(" "))

        # Now we are going to read in the post-processed mesh, that is a merge
        # of the partial view and of the completion from the CNN
        mesh = curvox.mesh_conversions.read_mesh_msg_from_ply_filepath(temp_completion_filepath)

        self._result.mesh = mesh
        
        self._as.set_succeeded(self._result)
        rospy.loginfo('Finished Msg')


if __name__ == "__main__":

    cnns = {
        "depth": {
            "cnn_python_module" : "shape_completion_server.trained_models.depth_y17_m05_d26_h14_m22_s35.models.reconstruction_network",
            "weights_filepath" : rospkg.RosPack().get_path('shape_completion_server')+'/scripts/shape_completion_server/trained_models/depth_y17_m05_d26_h14_m22_s35/results/best_weights.h5'
        }
    }

    rospy.init_node("mesh_completion_node")
    server = MeshCompletionServer(cnns)
    rospy.spin()
