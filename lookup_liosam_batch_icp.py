import argparse
import math
import numpy as np
import socket
import importlib
import os
import re
import pandas as pd
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import sys
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
# import pygicp
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
#import pcl
#from tools import *

cudnn.enabled = True

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_candidates = 10
Disco_FFT = []
Disco_data = []
Disco_label = []
yaw_diff_pc = []


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
    text = text.strip()
    text = text.replace(" ", ",")
    #print(text)
    str_list = eval(text)
    #print(str_list)
    pose_SE3=np.array(str_list)
    #pose_SE3 =  np.asarray(str , dtype = float)
    #pose_SE3 = np.vstack( (np.reshape(pose_SE3, (3, 4)), np.asarray([0,0,0,1])))
    #print(pose_SE3)
    return pose_SE3
    #poses.append(pose_SE3)

def getTransfromPose(pose):
    #print(pose)
    #tlist = [pose[3],pose[7],pose[11]]
    trans = [pose[3],pose[7],pose[11]]
    #print(trans)
    return trans

def getCandidate(pcdid):
    filename = cfg.SCAN_FOLDER + str(pcdid) + '.pcd'
    # print(filename)
    pcd = o3d.io.read_point_cloud(filename)
    submap_array = np.asarray(pcd.points)
    disco,FFT_candidate = infer_array(submap_array) 
    return FFT_candidate,submap_array


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
        # print('yaw_pc',yaw_pc) #单位是degree度
        yaw_diff_pc.append(yaw_pc)
        pred_angle_deg = yaw_pc # in degree
        pred_angle_rad = pred_angle_deg * np.pi / 180.
        init_pose_pc = getSE3(0, 0, pred_angle_rad)
        # print('init_pose_pc',init_pose_pc)
        rmse_pc = 0
        #fitness_pc, loop_transform = fast_gicp(scan_current, pc_candidate, max_correspondence_distance=cfg.icp_max_distance,fitness_score = cfg.icp_fitness_score, init_pose=init_pose_pc)
        fitness_pc,rmse_pc, loop_transform = o3d_icp(scan_current, pc_candidate, cfg.icp_max_distance, init_pose=init_pose_pc)
        fitness_list.append(fitness_pc)
        transform_list.append(loop_transform)
        rmse_list.append(rmse_pc)
        
        # print("\033[0;33;31m",'pred_angle_rad',pred_angle_rad,"\033[0m")
        # print("\033[0;33;31m",'yaw_pc',yaw_pc,'init_pose_pc',"\033[0m")

        #loop_transform = np.linalg.inv(loop_transform)
        # print('fitness score',fitness_pc)
        #print(loop_transform)
        #print('saving transformed map')
        # pcd = o3d.geometry.PointCloud()
        # pcd.points = o3d.utility.Vector3dVector(scan_current)
        # pcd.transform(init_pose_pc)
        # o3d.io.write_point_cloud("./aligned_scan/"+str(pcdid)+'-init'+'.pcd', pcd) #+'-init'
        if fitness_pc > cfg.icp_fitness_score:
            print("ICP fitness score is larger than threshold, accept the loop.")
            # print("loop_transform",loop_transform)
            # print('saving transformed map')
            # pcd = o3d.geometry.PointCloud()
            # pcd.points = o3d.utility.Vector3dVector(scan_current)
            # pcd.transform(loop_transform)
            # o3d.io.write_point_cloud("./32campus_mapping/aligned_scan/"+str(pcdid)+'-aligned'+'.pcd', pcd) #+'-init'
            # !!! convert the pointcloud transformation matrix to the frame transformation matrix
            #loop_transform = np.linalg.inv(loop_transform)
        else:
            print("DiSCO: ICP fitness score is less than threshold, reject the loop.")
    # print(fitness_list)
    # print(rmse_list)

    # print("got segmented map")
    # print( "\033[0;33;36m", "nearest pcd id is ",pcdid_list ,"\033[0m") #打印其pcd_id

    max_fitness = max(fitness_list) # 求列表最大值
    max_fitidx = fitness_list.index(max_fitness) # 求最大值对应索引
    min_rmse = min(rmse_list) 
    min_rmseidx = rmse_list.index(min_rmse) 

    return pcdid_list[max_fitidx],pcdid_list[min_rmseidx]

def getSegmentedMap(pcdid,map_read):
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
    # map_read = pcl.load('./voxmap_campus_2.pcd')
    min = np.zeros((1,3), dtype=np.float32)
    max = np.zeros((1,3), dtype=np.float32)
    map_array= map_read.to_array()
    x = map_array[:,0]
    y = map_array[:,1]
    z = map_array[:,2]

    
    #pc_candidate = o3d.io.read_point_cloud(this_pcd_name)
    submap_array = cloud_filtered_y.to_array()
    pc_candidate = submap_array
    # pcd = o3d.geometry.PointCloud()
    # pcd.points = o3d.utility.Vector3dVector(submap_array)
    # pc_candidate = np.asarray(pcd.points)

    disco,FFT_candidate = infer_array(submap_array) 
    return FFT_candidate,pc_candidate

def evaluate_query(scan_id,posefile,TfTree):
    pose = readPose_mine(scan_id,posefile)
    trans = getTransfromPose(pose)
    trans = np.array(trans)
    #dist_to_knn, idx_of_knn = TfTree.query(trans.reshape(1, -1),k=num_candidates)
    idx_of_knn = TfTree.query_radius(trans.reshape(1, -1), r=3.0)
    #print(dist_to_knn)
    final_idx = idx_of_knn[0]
    #print(len(idx_of_knn[0]))
    if(len(idx_of_knn[0]) == 0):
        dist_to_knn, idx_of_knn = TfTree.query(trans.reshape(1, -1),k=num_candidates)
        final_idx = idx_of_knn[0]
    elif(len(idx_of_knn[0]) >10):
        final_idx = idx_of_knn[0][0:10]
    print(final_idx)
    return final_idx

def constructTftree(lookupPoseFile):
    file = open(lookupPoseFile) 
    lines = len(file.readlines()) 
    Trans_data = []
    for tf_id in range(lines): 
        pose = readPose_mine(tf_id+1,lookupPoseFile)
        trans = getTransfromPose(pose)
        Trans_data.append(trans)
    #print(Trans_data)
    tftree = KDTree(Trans_data,metric='manhattan')
    return tftree

def QueryinTree(res_folder,frames_folder,lookupPoseFile,all_result_csv):
    print('QueryinTree')
    file_list = os.listdir(res_folder)
    #res_file = open(result_txt,"w")
    #file_list.sort()
    #print(file_list)
    times = 0
    TfTree = constructTftree(lookupPoseFile)
    #input()

    for file in file_list: 
        times = times+1
        #print(str(times)+ '/' + str(len(file_list)))
        (filename,extension) = os.path.splitext(file)
        #pcd_file = os.path.join(cfg.INPUT_FILEFOLDER, filename) + '.pcd'
        #print("Loading " , file)
        loadData = np.load(res_folder+file)
        id = (int)(filename)
        Disco_data.append(loadData[0])
        Disco_label.append(id)
        
    
    #print(Disco_data)
    tree = KDTree(Disco_data,metric='manhattan')
    query_result = []
    eval_result = []
    acc_result = []
    bestfit_result = []
    bestrmse_result = []
    intersec2_list = []
    intersec3_list = []
    acc_num = 0
    statistic_acc_rate = 0
    scan_list = os.listdir(frames_folder)
    start = 350
    end = 400
    for scan_id in range(start,end):  #len(scan_list)
        scan_filename = str(scan_id+1)+ '.pcd'
        print(scan_filename)
        # scan_read = pcl.load(frames_folder+scan_filename)
        # scan_array = scan_read.to_array()
        scan_read = o3d.io.read_point_cloud(frames_folder+scan_filename)
        scan_array = np.asarray(scan_read.points)
    
        disco,fft_current = infer_array(scan_array)
        out = disco.cpu().numpy()
    
        dist_to_knn, idx_of_knn = tree.query(out.reshape(1,-1),k=num_candidates)
        
        #print(idx_of_knn)   # k个近邻的索引
        nearest_pcd_id= [] 
        nearest_pcd_name = []
        for idx in idx_of_knn[0]:
            nearest_pcd_id.append (Disco_label[idx])
        query_result.append(nearest_pcd_id)
        print( "\033[0;33;36m", "query_kdtree_dis ",dist_to_knn ,"\033[0m") #打印其pcd_id
        print( "\033[0;33;36m", "nearest pcd id is ",nearest_pcd_id ,"\033[0m") #打印其pcd_id
        #nearest_pcd_id 存入txt
        nearest_pcd_id = np.array(nearest_pcd_id)
        truth_id = evaluate_query(scan_id+1,cfg.INPUT_POSES,TfTree)
        eval_result.append(truth_id)
        #查询有几个nearest_pcd_id在truth_id中
        intersec1 = []
        intersec1 = list(set(nearest_pcd_id).intersection(set(truth_id)))
        acc = len(intersec1) / num_candidates
        #a = np.array(a)
        print(acc,intersec1)
        acc_result.append(acc)
        if(acc>0):
            acc_num = acc_num+1
            statistic_acc_rate = statistic_acc_rate + acc
            #获取本次Scan的估计位姿
            best_fitness, best_rmse= getEstimateLocation(nearest_pcd_id,fft_current)
            print('best_fitness',best_fitness,'best_rmse',best_rmse)
            intersec2 = list(set(truth_id).intersection(set([best_fitness])))
            intersec3 = list(set(truth_id).intersection(set([best_rmse])))
            intersec2_list.append(len(intersec2))
            intersec3_list.append(len(intersec3))
            #下一步，三个result保存到.csv中
            bestfit_result.append(best_fitness)
            bestrmse_result.append(best_rmse)
        else:
            intersec2_list.append(0)
            intersec3_list.append(0)
            bestfit_result.append(0)
            bestrmse_result.append(0)

    statistic_num = end - start +1 # len(scan_list)
    #print(query_result)
    total_acc_rate = [statistic_acc_rate/statistic_num]
    recall_rate = [acc_num/statistic_num]
    print("\033[0;0;31m","acc_num",acc_num,"total_acc_rate",total_acc_rate,"recall_rate",recall_rate,"\033[0m")

    basic_frame = pd.DataFrame({'query_result':query_result,'eval_result':eval_result,'acc_rate':acc_result,'bestfit_result':bestfit_result,'bestrmse_result':bestrmse_result,'intersec2_list':intersec2_list,'intersec3_list':intersec3_list}) 
    statistic_frame = pd.DataFrame({'total_acc_rate':total_acc_rate,'recall_rate':recall_rate}) 
    dataframe = pd.concat([basic_frame,statistic_frame], axis=1)
    dataframe.to_csv(all_result_csv,index=False,sep=',')    

    print("result wrote to",all_result_csv)
    #query_result = np.array(query_result)
    # np.savetxt(query_result_txt, query_result, fmt='%d', delimiter=' ')
    # np.savetxt(eval_result_txt, eval_result, fmt='%d', delimiter=' ')
    
    return 
    #return nearest_pcd_id,fft_current


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--dimension', type=int, default=1024)
    parser.add_argument('--input_type', default='point',
                        help='Input of the network, can be [point] or scan [image], [default: point]')
    parser.add_argument('--max_icp_iter', type=int, default=50) # 20 iterations is usually enough
    parser.add_argument('--icp_tolerance', type=float, default=0.001) 
    parser.add_argument('--icp_max_distance', type=float, default=5.0)
    parser.add_argument('--icp_fitness_score', type=float, default=0.10) # icp fitness score threshold

    FLAGS = parser.parse_args()


    cfg.FEATURE_OUTPUT_DIM = FLAGS.dimension
    cfg.num_ring = 40
    cfg.num_sector = 120
    cfg.num_height = 20
    cfg.max_length = 1
    cfg.max_height = 1
    cfg.icp_max_distance = FLAGS.icp_max_distance
    cfg.max_icp_iter = FLAGS.max_icp_iter
    cfg.icp_tolerance = FLAGS.icp_tolerance
    cfg.icp_fitness_score = FLAGS.icp_fitness_score


    cfg.LOG_DIR = './log/'
    cfg.MODEL_FILENAME = "model.ckpt"
    cfg.INPUT_TYPE = FLAGS.input_type

    res_folder = "./16campus/results_liosam_1024_unet/"
    frames_folder = "./16campus/frames/"
    groundtruth = "./16campus/groundtruth/tf.txt"
    all_result_csv = "./16campus/result.csv"
    lookupPosefolder = "./16campus/tf/tf.txt"
    scan_folder = "./16campus/Scans/"

    cfg.INPUT_POSES = groundtruth  #'./query_scan2.pcd'
    cfg.FRAMES_FOLDER = frames_folder
    cfg.SCAN_FOLDER = scan_folder
    
    #Init()
    QueryinTree(res_folder,frames_folder,lookupPosefolder,all_result_csv)

    
    
    

    # fw = open("./pcdlist.txt", 'w') 
    # for pcdid in pcdid_list: 
    #     fw.write(str(pcdid)+'.pcd')  
    #     fw.write('\n')
    # fw.close()

    