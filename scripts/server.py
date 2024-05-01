#!/usr/bin/env python

# SPDX-FileCopyrightText: 2022 Hiroto Horimoto
# SPDX-License-Identifier: BSD-3-Clause

import rospy
from std_msgs.msg import Header
from sensor_msgs.msg import PointCloud2, PointField
import sensor_msgs.point_cloud2 as pc2
from geometry_msgs.msg import Pose2D
import os
from test_disco.srv import relocalize_pointcloud


def relocalize(req_pose):
    # pc = pc2.read_points(req_pc, skip_nans=True, field_names=("x", "y", "z"))
    # pc_list = []
    # for p in pc:
    #     pc_list.append([p[0],p[1],p[2]])
    print('req_pose',req_pose)
    re_pose = Pose2D(1,2,3)
    return re_pose

def relocalize_server():
    rospy.init_node('relocalize_node')
    s = rospy.Service('relocalize_srv', relocalize_pointcloud, relocalize)
    print('begin service')
    # spin() keeps Python from exiting until node is shutdown
    rospy.spin()


if __name__ == '__main__':
    relocalize_server()