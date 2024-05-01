import argparse
import math
import numpy as np
import socket
import importlib
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import sys
import time
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.backends import cudnn
import cv2
from sklearn.neighbors import NearestNeighbors
from sklearn.neighbors import KDTree
from scipy.spatial import distance
from torchvision import transforms, utils
from tools import *
import linecache
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
from loading_pointclouds_mine import *
import models.DiSCO as SC
from tensorboardX import SummaryWriter
import loss.loss_function
import gputransform
import config as cfg
import scipy.io as scio
from infer_mine import infer_array,infer
import re
import transforms3d as tr
import scipy.spatial.transform

cudnn.enabled = True

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_candidates = 5
Disco_FFT = []
Disco_data = []
Disco_label = []
yaw_diff_pc = []
global map_read
global x_num
global y_num
global min_x 
global max_x 
global min_y 
global max_y 
global min_z 
global max_z 
global square_size
global square_gap
square_size = 120.0
square_gap = 2.0

def Init():
    global x_num
    global y_num
    global min_x 
    global max_x 
    global min_y 
    global max_y 
    global min_z 
    global max_z 
    global square_size
    global square_gap

    map_read = o3d.io.read_point_cloud('./voxmap_campus_2.pcd')
    map_array = np.asarray(map_read.points)
    #确定分辨率
    w = map_read.width 
    h = map_read.height
    print("width " , w ," height " , h )
    #划分体素格子
    min = np.zeros((1,3), dtype=np.float32)
    max = np.zeros((1,3), dtype=np.float32)
    #map_array= map_read.to_array()
    x = map_array[:,0]
    y = map_array[:,1]
    z = map_array[:,2]
    min_x = np.min(x)
    max_x = np.max(x)
    min_y = np.min(y)
    max_y = np.max(y)
    min_z = np.min(z)
    max_z = np.max(z)
    print("original map ")
    print("->min_x = ", min_x)
    print("->max_x = ", max_x)
    print("->min_y = ", min_y)
    print("->max_y = ", max_y)
    print("->min_z = ", min_z)
    print("->max_z = ", max_z)

    x_num = math.ceil(((max_x-min_x) - square_size)/square_gap) +1;
    y_num = math.ceil(((max_y-min_y) - square_size)/square_gap) +1;
    print( "\033[0;33;36m", 'x_num',x_num,'y_num',y_num ,"\033[0m")

def get_line_context(file_path, line_number):
    return linecache.getline(file_path, line_number).strip()

def readPose_liosam(fileid,posefile):
    poses = []
    line = int(fileid) +1
    text = get_line_context(posefile,line)
    text = text.replace(" ", ",")
    #print(text)
    str_list = eval(text)
    #print(str_list)
    pose_SE3=np.array(str_list)
    #pose_SE3 =  np.asarray(str , dtype = float)
    pose_SE3 = np.vstack( (np.reshape(pose_SE3, (3, 4)), np.asarray([0,0,0,1])))
    #print(pose_SE3)
    return pose_SE3
    #poses.append(pose_SE3)

def readPose_mine(tfid,posefile):
    poses = []
    line = int(tfid)
    text = get_line_context(posefile,line)
    text = re.sub('\s+', ' ',text)
    text = text.replace(" ", ",")
    #print(text)
    str_list = eval(text)
    #print(str_list)
    pose_SE4=np.array(str_list)
    #pose_SE3 =  np.asarray(str , dtype = float)
    #pose_SE3 = np.vstack( (np.reshape(pose_SE3, (3, 4)), np.asarray([0,0,0,1])))
    #print(pose_SE3)
    return pose_SE4
    #poses.append(pose_SE3)

def getTransfromPose(pose):
    #print(pose)
    #tlist = [pose[3],pose[7],pose[11]]
    trans = [pose[3],pose[7]] # #只需要trans的前两个，即XY
    #print(trans)
    return trans

def getYawfromPose(pose):
    pitch, roll, yaw = scipy.spatial.transform.Rotation.from_matrix(pose).as_euler('xyz')
    #print(pose)
    #z = math.atan2(pose[1,0], pose[0,0])
    # print(pitch, roll, yaw)
    return yaw

def getEstimateLocation(pcdid_list,fft_current):
    # map_read = o3d.io.read_point_cloud('./voxmap_campus_2.pcd')
    # map_array = np.asarray(map_read.points)
    corr2soft = SC.Corr2Softmax(200., 0.)
    corr2soft = corr2soft.to(device)

    # scan_read = pcl.load(cfg.INPUT_FILENAME)
    # sor = scan_read.make_voxel_grid_filter()
    # sor.set_leaf_size(0.1, 0.1, 0.1)
    # scan_filtered = sor.filter()
    # scan_current= scan_filtered.to_array()
    # convert the point cloud to o3d point cloud

    # pcd = o3d.io.read_point_cloud(cfg.INPUT_FILENAME)
    #pcd = pcd.voxel_down_sample(voxel_size=0.2)
    #scan_current = np.asarray(pcd.points)
    
    # scan_current = o3d.io.read_point_cloud(cfg.INPUT_FILENAME)
    # pcd = o3d.geometry.PointCloud()
    # pcd.points = o3d.utility.Vector3dVector(xyz)
    # scan_current = np.asarray(pcd.points)
    #scan_current = pcd.points
    #print(scan_current)
    print(pcdid_list)
    fitness_list = []
    transform_list = []
    rmse_list = []
    for pcdid in pcdid_list: 
        #pose = readPose(pcdid)
        filename = cfg.SCAN_FOLDER + str(pcdid) + '.pcd'
        pcd = o3d.io.read_point_cloud(filename)
        scan_current = np.asarray(pcd.points)

        # pcd = o3d.geometry.PointCloud()
        # pcd.points = o3d.utility.Vector3dVector(scan_current)
        # 通过变换增加一个大的绿色点表示位置
        
        #pcd.transform(pose)
        #o3d.io.write_point_cloud("./32campus_mapping/aligned_scan/"+str(pcdid)+'-init'+'.pcd', pcd) #+'-init'

        FFT_candidate,pc_candidate = getCandidate(pcdid)

        #print(pc_candidate)
        yaw_pc, _ = phase_corr(FFT_candidate, fft_current, device, corr2soft)
 
        #print(yaw_pc)
        yaw_pc = (yaw_pc - cfg.num_sector//2) / float(cfg.num_sector) * 360.  # //是取整除 - 返回商的整数部分（向下取整）
        print('yaw_pc',yaw_pc) #单位是degree度
        yaw_diff_pc.append(yaw_pc)
        pred_angle_deg = yaw_pc # in degree
        pred_angle_rad = pred_angle_deg * np.pi / 180.
        init_pose_pc = getSE3(0, 0, pred_angle_rad)
        # print('init_pose_pc',init_pose_pc)
        rmse_pc= 0
        #fitness_pc, loop_transform = fast_gicp(scan_current, pc_candidate, max_correspondence_distance=cfg.icp_max_distance,fitness_score = cfg.icp_fitness_score, init_pose=init_pose_pc)
        fitness_pc,rmse_pc, loop_transform = o3d_icp(scan_current, pc_candidate, cfg.icp_max_distance, init_pose=init_pose_pc)
        fitness_list.append(fitness_pc)
        transform_list.append(loop_transform)
        rmse_list.append(rmse_pc)
        
        print("\033[0;33;31m",'pred_angle_rad',pred_angle_rad,"\033[0m")
        print("\033[0;33;31m",'yaw_pc',yaw_pc,'init_pose_pc',"\033[0m")
        # print(init_pose_pc)
        #loop_transform = np.linalg.inv(loop_transform)
        print('fitness score',fitness_pc)
        # print(loop_transform)
        if fitness_pc > cfg.icp_fitness_score:
            print("ICP fitness score is larger than threshold, accept the loop.")
            print("loop_transform")
            print(loop_transform)
            print('saving transformed map')
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(scan_current)
            pcd.transform(loop_transform)
            o3d.io.write_point_cloud("./16campus/aligned_scan/"+str(pcdid)+'-aligned'+'.pcd', pcd) #+'-init'
            # !!! convert the pointcloud transformation matrix to the frame transformation matrix
            #loop_transform = np.linalg.inv(loop_transform)
        else:
            print("DiSCO: ICP fitness score is less than threshold, reject the loop.")
    print(fitness_list)
    print(rmse_list)
    print("got segmented map")
    print( "\033[0;33;36m", "nearest pcd id is ",pcdid_list ,"\033[0m") #打印其pcd_id
    max_fitness = max(fitness_list) # 求列表最大值
    max_fitidx = fitness_list.index(max_fitness) # 求最大值对应索引
    # min_rmse = min(rmse_list) 
    # min_rmseidx = rmse_list.index(min_rmse) 

    return pcdid_list[max_fitidx],loop_transform


def getCandidate(pcdid):
    filename = cfg.SCAN_FOLDER + str(pcdid) + '.pcd'
    pcd = o3d.io.read_point_cloud(filename)
    submap_array = np.asarray(pcd.points)
    disco,FFT_candidate = infer_array(submap_array) 
    return FFT_candidate,submap_array


def constructTftree(lookupPoseFile):
    file = open(lookupPoseFile) 
    lines = len(file.readlines()) 
    Trans_data = []
    for tf_id in range(lines): #0 - len()-1
        pose = readPose_mine(tf_id+1,lookupPoseFile)
        trans = getTransfromPose(pose)
        Trans_data.append(trans) 
    #print(Trans_data)
    tftree = KDTree(Trans_data,metric='manhattan')
    return tftree

#通过某个trigger触发，同时传入pose，用于初始的定位
def getPossibleRange(lookupPoseFile,trans):
    TfTree = constructTftree(lookupPoseFile)
    #trans = getTransfromPose(pose)
    # trans = [XYyaw[0],XYyaw[1]]
    trans = np.array(trans)
    print(trans)
    idx_of_knn = TfTree.query_radius(trans.reshape(1, -1), r=20.0)
    print('possible range',idx_of_knn)
    return idx_of_knn[0]

def QueryinTree(res_folder,queryRange):
    print('QueryinTree')
    #times = time.time()
    #file_list = os.listdir(res_folder)
    cnts = 0
    
    for file_id in queryRange: 
        cnts = cnts+1
        #print(str(cnts)+ '/' + str(len(file_list)))
        #pcd_file = os.path.join(cfg.INPUT_FILEFOLDER, filename) + '.pcd'
        #print("Loading " , file)
        loadData = np.load(res_folder+str(file_id+1)+'.npy')
        Disco_data.append(loadData[0])
        Disco_label.append(file_id)
        
    
    #print(Disco_data)
    tree = KDTree(Disco_data,metric='manhattan')
    scan_read = o3d.io.read_point_cloud(cfg.INPUT_FILENAME)
    scan_array = np.asarray(scan_read.points)
    
    disco,fft_current = infer_array(scan_array)
    out = disco.cpu().numpy()
    
    dist_to_knn, idx_of_knn = tree.query(out.reshape(1,-1),k=num_candidates)
    print( "\033[0;33;36m", "query_kdtree_dis ",dist_to_knn ,"\033[0m") #打印其pcd_id
    #print(idx_of_knn)   # k个近邻的索引
    nearest_pcd_id= [] 
    bestfit_result = []
    nearest_pcd_name = []
    for idx in idx_of_knn[0]:
        nearest_pcd_id.append (Disco_label[idx])
    print( "\033[0;33;36m", "nearest pcd id is ",nearest_pcd_id ,"\033[0m") #打印其pcd_id
    #timee = time.time()
    #print("Process time:", timee - times, 's')
    #print(nearest_pcd_name)
    best_fitness_idx, loop_transform= getEstimateLocation(nearest_pcd_id,fft_current)
    print('best_fitness_idx',best_fitness_idx)
    # 找到最佳位置的tf
    tf_pose = readPose_mine(best_fitness_idx,lookupPoseFile)
    tf_pose = tf_pose.reshape(4,4)
    t1,r1,_,_ =tr.affines.decompose(tf_pose)
    t2,r2,_,_ =tr.affines.decompose(loop_transform)
    T = t1+t2
    r = scipy.spatial.transform.Rotation.from_matrix(r1) * scipy.spatial.transform.Rotation.from_matrix(r2)
    new_r = scipy.spatial.transform.Rotation.as_matrix(r)
    yaw = getYawfromPose(new_r)
    print(new_r)
    pose2D = [T[0],T[1],yaw]
    print(pose2D)
    # now_pose = tf_pose * loop_transform
    return pose2D



if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--input_poses', default='./16campus/tf/tf.txt',
                        help='input pose file name [default: ./yuanqu_data/other/optimized_poses.txt]')
    parser.add_argument('--input_filename', default='./query.pcd',
                        help='input file name [default: ./query.pcd]')
    parser.add_argument('--dimension', type=int, default=1024)
    parser.add_argument('--input_type', default='point',
                        help='Input of the network, can be [point] or scan [image], [default: point]')
    parser.add_argument('--max_icp_iter', type=int, default=50) # 20 iterations is usually enough
    parser.add_argument('--icp_tolerance', type=float, default=0.001) 
    parser.add_argument('--icp_max_distance', type=float, default=0.1)
    parser.add_argument('--icp_fitness_score', type=float, default=0.10) # icp fitness score threshold

    FLAGS = parser.parse_args()

    cfg.INPUT_POSES = FLAGS.input_poses  #'./query_scan2.pcd'
    cfg.INPUT_FILENAME = FLAGS.input_filename  #'./query_scan2.pcd'
    cfg.FEATURE_OUTPUT_DIM = FLAGS.dimension
    cfg.num_ring = 40
    cfg.num_sector = 120
    cfg.num_height = 20
    cfg.max_length = 1
    cfg.max_height = 1
    # cfg.icp_max_distance = FLAGS.icp_max_distance
    # cfg.max_icp_iter = FLAGS.max_icp_iter
    # cfg.icp_tolerance = FLAGS.icp_tolerance
    # cfg.icp_fitness_score = FLAGS.icp_fitness_score


    cfg.LOG_DIR = './log/'
    cfg.MODEL_FILENAME = "model.ckpt"
    cfg.INPUT_TYPE = FLAGS.input_type
    cfg.GLOBAL_MAP = './voxmap_campus_2.pcd'

    res_folder = "./16campus/results_liosam_1024_unet/"
    frames_folder = "./16campus/frames/"
    groundtruth = "./16campus/groundtruth/tf.txt"
    all_result_csv = "./16campus/result.csv"
    lookupPoseFile = "./16campus/tf/tf.txt"
    scan_folder = "./16campus/Scans/"

    cfg.INPUT_POSES = groundtruth  #'./query_scan2.pcd'
    cfg.INPUT_FILENAME = FLAGS.input_filename  #'./query_scan2.pcd'
    cfg.FRAMES_FOLDER = frames_folder
    cfg.SCAN_FOLDER = scan_folder

    #Init()
    test_pose = readPose_mine(90,lookupPoseFile)
    trans = getTransfromPose(test_pose)
    pose_SE3 = np.reshape(test_pose, (-1, 4))
    pose_SE3 = pose_SE3[0:3,0:3]
    R = pose_SE3
    yaw = getYawfromPose(R)
    test_pose2D = [ trans[0],trans[1],yaw]
    print('test_pose2D',test_pose2D)
    queryRange = getPossibleRange(lookupPoseFile,trans)
    now_pose = QueryinTree(res_folder,queryRange)
    
    

    # fw = open("./pcdlist.txt", 'w') 
    # for pcdid in pcdid_list: 
    #     fw.write(str(pcdid)+'.pcd')  
    #     fw.write('\n')
    # fw.close()

    