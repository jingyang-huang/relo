import argparse
import math
import numpy as np
import socket
import importlib
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import sys
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.backends import cudnn
import cv2
from sklearn.neighbors import NearestNeighbors
from sklearn.neighbors import KDTree
from torchvision import transforms, utils

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
from loading_pointclouds_mine import *
import models.DiSCO as SC
from tensorboardX import SummaryWriter
import loss.loss_function
import gputransform
import config as cfg
import scipy.io as scio
from infer_mine import infer_array
import open3d as o3d
import shutil
def infer_frame(input_folder):
    if(os.path.exists(cfg.RES_FOLDER)):
        shutil.rmtree(cfg.RES_FOLDER)
    os.mkdir(cfg.RES_FOLDER)
    files = os.listdir( input_folder )
    cnt = 0
    # 输出所有文件和文件夹
    for file in files:
        #print (file)
        cnt = cnt+ 1;
        frame_read = o3d.io.read_point_cloud(input_folder + file)
        frame_array = np.asarray(frame_read.points)

        print("cnt",cnt,"/",len(files) )
        #this_pcd_name = "./tools/test/"+ str(this_idx)+ ".pcd"
        #pcl.save(cloud_filtered_y,this_pcd_name) #不需要保存了
        disco,_ = infer_array(frame_array) 
        # filename = str(this_idx)
        # #save_res = os.path.join(os.path.join("./results/", filename),".npy")
        file_name,_ = os.path.splitext(file)
        disco_save_res = cfg.RES_FOLDER + '{}.npy'.format(file_name)
        # fft_save_res = './fft_results/{}.npy'.format(str(this_idx))
        #print(disco_save_res)
        np.save(disco_save_res, disco.cpu().numpy())
        # np.save(fft_save_res, fft_result.cpu().numpy())
        #out = disco.cpu().numpy()


    # #确定分辨率
    # w = map_read.width 
    # h = map_read.height
    # print("width " , w ," height " , h )
    # #划分体素格子
    # min = np.zeros((1,3), dtype=np.float32)
    # max = np.zeros((1,3), dtype=np.float32)
    # map_array= map_read.to_array()
    # x = map_array[:,0]
    # y = map_array[:,1]
    # z = map_array[:,2]
    # min_x = np.min(x)
    # max_x = np.max(x)
    # min_y = np.min(y)
    # max_y = np.max(y)
    # min_z = np.min(z)
    # max_z = np.max(z)
    # print("original map ")
    # print("->min_x = ", min_x)
    # print("->max_x = ", max_x)
    # print("->min_y = ", min_y)
    # print("->max_y = ", max_y)
    # print("->min_z = ", min_z)
    # print("->max_z = ", max_z)

    # times = 0
    # square_size = 120.0
    # square_gap = 2.0
    # x_num = math.ceil(((max_x-min_x) - square_size)/square_gap) +1;
    # y_num = math.ceil(((max_y-min_y) - square_size)/square_gap) +1;
    # #x_num = 1
    # #y_num = 1
    # print("\033[0;33;34m","creating a submap of ",x_num,"*",y_num," with square_size=",square_size ,"m and square_gap=",square_gap,"m","\033[0m")

    # for x_idx in range(x_num) :
    #     print("\033[0;33;32m","x is ",x_idx,"/",x_num-1,"\033[0m")
    #     for y_idx in range(y_num) :
    #         this_idx = y_idx + x_idx*y_num
    #         this_x_left = min_x + square_gap*x_idx
    #         this_x_right = min_x + square_gap*x_idx + square_size
    #         this_y_left = min_y + square_gap*y_idx
    #         this_y_right = min_y + square_gap*y_idx + square_size
    #         passthrough_x = map_read.make_passthrough_filter()
    #         passthrough_x.set_filter_field_name("x")
    #         passthrough_x.set_filter_limits(this_x_left, this_x_right)
    #         cloud_filtered_x = passthrough_x.filter()

    #         passthrough_y = cloud_filtered_x.make_passthrough_filter()
    #         passthrough_y.set_filter_field_name("y")
    #         passthrough_y.set_filter_limits(this_y_left, this_y_right)
    #         cloud_filtered_y = passthrough_y.filter()
    #         if(cloud_filtered_y.size > 20):
    #             print("this_idx",this_idx,"/",x_num*y_num)
    #             #this_pcd_name = "./tools/test/"+ str(this_idx)+ ".pcd"
    #             #pcl.save(cloud_filtered_y,this_pcd_name) #不需要保存了
    #             submap_array = cloud_filtered_y.to_array()
    #             disco,_ = infer_array(submap_array) 
    #             # filename = str(this_idx)
    #             # #save_res = os.path.join(os.path.join("./results/", filename),".npy")
    #             disco_save_res = './results/{}.npy'.format(str(this_idx))
    #             # fft_save_res = './fft_results/{}.npy'.format(str(this_idx))
    #             #print(save_res)
    #             np.save(disco_save_res, disco.cpu().numpy())
    #             # np.save(fft_save_res, fft_result.cpu().numpy())
    #             #out = disco.cpu().numpy()


                

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--dimension', type=int, default=1024)
    FLAGS = parser.parse_args()

    
    cfg.FEATURE_OUTPUT_DIM = FLAGS.dimension
    cfg.num_ring = 40
    cfg.num_sector = 120
    cfg.num_height = 20
    cfg.max_length = 1

    cfg.LOG_DIR = './log/'
    cfg.MODEL_FILENAME = "model.ckpt"
    cfg.SCAN_FOLDER= './querys/Scans/'
    cfg.RES_FOLDER= "./querys/results_scans_1024/"
    infer_frame(cfg.SCAN_FOLDER)