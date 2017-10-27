#!/usr/bin/env python

import importlib
import numpy as np
import subprocess
import tempfile
import argparse
import os

import pcl
import binvox_rw

import rospkg
import actionlib
import rospy

import pc_pipeline_msgs.msg
import pc_object_completion_cnn.srv
from sensor_msgs import point_cloud2

import curvox.pc_vox_utils
import curvox.mesh_conversions
import curvox.cloud_conversions


class MeshCompletionServer(object):
    def __init__(self, ns, cnns, flip_batch_x):

        rospy.loginfo("Starting Completion Server")

        self.flip_batch_x = flip_batch_x
        self.patch_size = 40
        self.cnns = cnns
        self.cnn_python_module = cnns[ns]["cnn_python_module"]
        self.weights_filepath = cnns[ns]["weights_filepath"]

        py_module = importlib.import_module(self.cnn_python_module)
        global model
        model = py_module.get_model()
        model.load_weights(self.weights_filepath)
        model._make_predict_function()

        self.post_process_executable = "mesh_reconstruction"

        self._feedback = pc_pipeline_msgs.msg.CompletePartialCloudFeedback()
        self._result = pc_pipeline_msgs.msg.CompletePartialCloudResult()

        self._as = actionlib.SimpleActionServer(
            ns + "/object_completion",
            pc_pipeline_msgs.msg.CompletePartialCloudAction,
            execute_cb=self.completion_cb,
            auto_start=False)

        self._switch_cnn_type_srv = rospy.Service(
            ns + '/shape_completion_server/set_cnn_type',
            pc_object_completion_cnn.srv.SetCNNType, self.set_cnn_type)

        self._as.start()

        rospy.loginfo("Started Completion Server")

    def complete_voxel_grid(self, batch_x_B012C):

        # The new version of keras takes the data as B012C
        # NOT BZCXY so we do not need to transpose it.
        batch_x = batch_x_B012C

        # run the batch through the network and get the completion
        # pred is actually flat, because we do not have a reshape as the final
        # layer of the net, so pred's shape is something like
        # (batch_size, 40*40*40)
        global model
        pred = model.predict(batch_x)

        # The new version of keras takes the data as B012C
        # NOT BZCXY so we do not need to transpose it.
        pred_as_b012c = pred.reshape(1, self.patch_size, self.patch_size,
                                     self.patch_size, 1)
        completed_region = pred_as_b012c[0, :, :, :, 0]

        return completed_region

    def set_cnn_type(self, request):
        cnn_name = request.cnn_name
        if cnn_name not in self.cnns:
            return shape_completion_server.srv.SetCNNTypeResponse(
                success=False)

        self.cnn_python_module = cnns[cnn_name]["cnn_python_module"]
        self.weights_filepath = cnns[cnn_name]["weights_filepath"]

        py_module = importlib.import_module(self.cnn_python_module)
        global model
        model = py_module.get_model()
        model.load_weights(self.weights_filepath)

        return shape_completion_server.srv.SetCNNTypeResponse(success=True)

    def completion_cb(self, goal):
        rospy.loginfo('Received Completion Goal')

        self._feedback = pc_pipeline_msgs.msg.CompletePartialCloudFeedback()
        self._result = pc_pipeline_msgs.msg.CompletePartialCloudResult()

        temp_pcd_handle, temp_pcd_filepath = tempfile.mkstemp(suffix=".pcd")
        os.close(temp_pcd_handle)
        partial_pc_np = curvox.cloud_conversions.cloud_msg_to_np(
            goal.partial_cloud)
        pcd = curvox.cloud_conversions.np_to_pcl(partial_pc_np)
        pcl.save(pcd, temp_pcd_filepath)

        partial_vox = curvox.pc_vox_utils.pc_to_binvox_for_shape_completion(
            points=partial_pc_np[:, 0:3], patch_size=self.patch_size)

        batch_x = np.zeros(
            (1, self.patch_size, self.patch_size, self.patch_size, 1),
            dtype=np.float32)
        batch_x[0, :, :, :, 0] = partial_vox.data

        
        if self.flip_batch_x:
            rospy.loginfo("Flipping Batch X, if performance is poor,\
            try setting flip_batch_x=False")
            batch_x = batch_x.transpose(0, 2, 1, 3, 4)
            batch_x_new = np.zeros_like(batch_x)
            for i in range(40):
                batch_x_new[0, i, :, :, 0] = batch_x[0, 40 - i - 1, :, :, 0]

            batch_x = batch_x_new
        else:
            rospy.loginfo(
                "NOT Flipping Batch X, if performance is poor,\
                try setting flip_batch_x=True"
            )

        # output is the completed voxel grid,
        # it is all floats between 0,1 as last layer is softmax
        # think of this as probability of occupancy per voxel
        # output.shape = (X,Y,Z)
        output = self.complete_voxel_grid(batch_x)

        if self.flip_batch_x:
            rospy.loginfo("flipping batch x back")
            output_new = np.zeros_like(output)
            for i in range(40):
                output_new[i, :, :] = output[40 - i - 1, :, :]

            output = output_new
            output = output.transpose(1, 0, 2)

        # mask the output, so above 0.5 is occupied
        # below 0.5 is empty space.
        output_vox = np.array(output) > 0.5

        # Save the binary voxel grid as an occupancy map
        # in a binvox file
        completed_vox = binvox_rw.Voxels(
            output_vox, partial_vox.dims, partial_vox.translate,
            partial_vox.scale, partial_vox.axis_order)

        # Now we save the binvox file so that it can be passed to the
        # post processing along with the partial.pcd
        temp_handle, temp_binvox_filepath = tempfile.mkstemp(
            suffix="output.binvox")
        os.close(temp_handle)
        binvox_rw.write(completed_vox, open(temp_binvox_filepath, 'w'))

        # Now we save the binvox file so that it can be passed to the
        # post processing along with the partial.pcd
        temp_handle, temp_input_binvox_file = tempfile.mkstemp(
            suffix="input.binvox")
        os.close(temp_handle)
        binvox_rw.write(partial_vox, open(temp_input_binvox_file, 'w'))

        # This is the file that the post-processed mesh will be saved it.
        temp_handle, temp_completion_filepath = tempfile.mkstemp(suffix=".ply")
        os.close(temp_handle)
        # This command will look something like
        # mesh_reconstruction tmp/completion.binvox tmp/partial.pcd tmp/post_processed.ply
        cmd_str = self.post_process_executable + " " + temp_binvox_filepath + " " + temp_pcd_filepath \
                  + " " + temp_completion_filepath + " --cuda"

        subprocess.call(cmd_str.split(" "))

        # Now we are going to read in the post-processed mesh, that is a merge
        # of the partial view and of the completion from the CNN
        mesh = curvox.mesh_conversions.read_mesh_msg_from_ply_filepath(
            temp_completion_filepath)

        self._result.mesh = mesh

        self._as.set_succeeded(self._result)
        rospy.loginfo('Finished Msg')


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Complete a partial object view")
    parser.add_argument(
        "ns",
        type=str,
        help=
        "Namespace used to create action server, also determines what model to load.  Ex: depth, depth_and_tactile"
    )
    parser.add_argument(
        "--flip_batch_x",
        type=bool,
        default=True,
        help=
        "Z+ should be extending away from the camera, sometime binvox files have this as Y+ and need to be rotated. "
    )
    args = parser.parse_args()
    cnns = {
        "depth": {
            "cnn_python_module":
            "shape_completion_server.trained_models.depth_y17_m05_d26_h14_m22_s35_bare_keras_v2.reconstruction_network",
            "weights_filepath":
            rospkg.RosPack().get_path('pc_object_completion_cnn') +
            '/scripts/shape_completion_server/trained_models/depth_y17_m05_d26_h14_m22_s35_bare_keras_v2/best_weights.h5'
        },
        "depth_and_tactile": {
            "cnn_python_module":
            "shape_completion_server.trained_models.depth_and_tactile_y17_m08_d09_h15_m55_s53_bare_keras_v2.reconstruction_network",
            "weights_filepath":
            rospkg.RosPack().get_path('pc_object_completion_cnn') +
            '/scripts/shape_completion_server/trained_models/depth_and_tactile_y17_m08_d09_h15_m55_s53_bare_keras_v2/best_weights.h5'
        }
    }

    rospy.init_node(args.ns + "mesh_completion_node")
    server = MeshCompletionServer(args.ns, cnns, args.flip_batch_x)
    rospy.spin()
