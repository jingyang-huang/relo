#!/usr/bin/env python

# SPDX-FileCopyrightText: 2022 Hiroto Horimoto
# SPDX-License-Identifier: BSD-3-Clause

import rospy
from relocalization_disco.srv import relocalize_pointcloud
from geometry_msgs.msg import Pose2D
import tf
import math
from scipy.spatial.transform import Rotation as R

def quaternion2euler(quaternion):
    r = R.from_quat(quaternion)
    euler = r.as_euler('xyz', degrees=True)
    return euler

def test_localize_srv(last_pose):
    rospy.wait_for_service('relocalize_srv')
    now_pose = None
    try:
        relocalize_handler = rospy.ServiceProxy('relocalize_srv', relocalize_pointcloud)#先是服务名称，然后是消息类型
        now_pose = relocalize_handler(last_pose)
        if now_pose:
            rospy.loginfo("Run test succeeded")
            return now_pose
    except rospy.ServiceException:
        rospy.loginfo("Run test failed")
        
def main():
    rospy.init_node('client')
    rospy.loginfo("Start relocalize test")
    rospy.sleep(1)
    # test_last_pose = Pose2D(4.07623,-2.90824,-1.57369140527878)
    # #print(test_last_pose.x)
    # print(test_last_pose)
    

    listener = tf.TransformListener()
    get_pose = False
    while not rospy.is_shutdown() and not get_pose:
      try:
        now = rospy.Time.now()
        listener.waitForTransform('/map', '/velodyne', rospy.Time(0), rospy.Duration(0.1))
        (trans, rot) = listener.lookupTransform('/map', '/velodyne', rospy.Time(0))
        # print(trans[0], trans[1], trans[2], rot[0], rot[1], rot[2], rot[3])
        print('trans',trans)
        get_pose = True
      except (tf.Exception, tf.LookupException, tf.ConnectivityException):
        print("tf echo error")
        continue
      rospy.sleep(1)
    euler = quaternion2euler(rot)
    print('euler',euler)
    yaw = euler[2] / 180 *math.pi
    last_pose = Pose2D(trans[0], trans[1], yaw)
    print('last_pose',last_pose)
    now_pose = test_localize_srv(last_pose)
    print('now_pose',now_pose)


if __name__ == "__main__":
    main()