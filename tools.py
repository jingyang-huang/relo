#!/home/client/miniconda3/envs/py3-mink/bin/python3
import os
import cv2
import sys
import math
import time
# import rospy
import torch
import socket
import gputransform
import argparse
import importlib
import numpy as np
import config as cfg
import scipy.io as scio
import loss.loss_function
#import pygicp
import torch.nn as nn
import open3d as o3d
import models.DiSCO as SC
import matplotlib.pyplot as plt
# import sensor_msgs.point_cloud2 as pc2
from torch.backends import cudnn
from torchvision import transforms, utils
# from geometry_msgs.msg import Pose
# from sensor_msgs.msg import PointCloud2
from sklearn.neighbors import NearestNeighbors, KDTree
# from tf.transformations import translation_matrix, quaternion_matrix, translation_from_matrix, quaternion_from_matrix
# import pcl

# def pcl_icp(source,target,tolerance=0.2, init_pose=np.eye(4)):
#     icp = source.make_IterativeClosestPoint()
#     icp.setMaximumIterations (cfg.max_icp_iter)
#     icp.setRelativeMSE (cfg.icp_tolerance)
#     # Final = icp.align()
#     converged, transf, estimate, fitness = icp.icp(source, target)
#     return fitness,transf

# achieve point-to-point icp with open3d
def o3d_icp(source_array, target_array, tolerance=0.2, init_pose=np.eye(4)):
    source = o3d.geometry.PointCloud()
    source.points = o3d.utility.Vector3dVector(source_array)
    target = o3d.geometry.PointCloud()
    target.points = o3d.utility.Vector3dVector(target_array)

    # source = source.voxel_down_sample(voxel_size = 0.02)
    # target = target.voxel_down_sample(voxel_size = 0.02)

    # o3d.visualization.draw_geometries([source])
    # o3d.visualization.draw_geometries([target])

    # apply outlier removal
    source.remove_statistical_outlier(nb_neighbors=20, std_ratio=1.0)
    target.remove_statistical_outlier(nb_neighbors=20, std_ratio=1.0)

    # run icp
    # result = o3d.pipelines.registration.registration_icp(source, target, tolerance, init_pose,
    #                o3d.pipelines.registration.TransformationEstimationPointToPoint(),
    #                o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=2000))

    result = o3d.pipelines.registration.registration_icp(
        source, target, tolerance, init_pose, 
        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=50))

    # result = o3d.treg.icp(source, target, tolerance, init_pose,
    #                 o3d.pipelines.registration.TransformationEstimationPointToPoint())
    #print(result)
    # get the icp fitness score
    fitness = result.fitness
    rmse = result.inlier_rmse
    # get the transformation matrix
    T_matrix = result.transformation

    return fitness,rmse, T_matrix


# apply icp using fast_gicp (https://github.com/SMRT-AIST/fast_gicp)
def fast_gicp(source, target, max_correspondence_distance=1.0,fitness_score=1.0, init_pose=np.eye(4)):
    # downsample the point cloud before registration

    # source = pygicp.downsample(source, 0.2)
    # target = pygicp.downsample(target, 0.2)

    
    # pygicp.FastGICP has more or less the same interfaces as the C++ version
    # method 1 
    # gicp = pygicp.FastGICP()
    # gicp.set_input_target(target)
    # gicp.set_input_source(source)
    # # optional arguments
    # gicp.set_num_threads(4)
    # gicp.set_max_correspondence_distance(max_correspondence_distance)

    # method 2
    gicp = pygicp.FastVGICP()
    gicp.set_input_target(target)
    gicp.set_input_source(source)
    # optional arguments
    gicp.set_resolution(1.0)
    #gicp.set_neighbor_search_method(max_correspondence_distance)


    # align the point cloud using the initial pose calculated by DiSCO
    T_matrix = gicp.align(initial_guess=init_pose)

    # get the fitness score
    fitness = gicp.get_fitness_score(fitness_score)
    #fitness = 0
    # get the transformation matrix
    T_matrix = gicp.get_final_transformation()




    return fitness, T_matrix
    
# get the rotation matrix from euler angles
def euler2rot(roll, pitch, yaw):
    R_x = np.array([[1, 0, 0],
                    [0, np.cos(roll), -np.sin(roll)],
                    [0, np.sin(roll), np.cos(roll)]])
    R_y = np.array([[np.cos(pitch), 0, np.sin(pitch)],
                    [0, 1, 0],
                    [-np.sin(pitch), 0, np.cos(pitch)]])
    R_z = np.array([[np.cos(yaw), -np.sin(yaw), 0],
                    [np.sin(yaw), np.cos(yaw), 0],
                    [0, 0, 1]])
    R = np.dot(R_z, np.dot(R_y, R_x))

    return R


# get the SE3 rotation matrix from the x, y translation and yaw angle    
def getSE3(x, y, yaw):
    R = np.eye(4)
    R[:3, :3] = euler2rot(0, 0, yaw)
    R[:3, 3] = np.array([x, y, 0])

    return R

def fftshift2d(x):
    for dim in range(1, len(x.size())):
        n_shift = x.size(dim)//2
        if x.size(dim) % 2 != 0:
            n_shift = n_shift + 1  # for odd-sized images
        x = roll_n(x, axis=dim, n=n_shift)
    return x  # last dim=2 (real&imag)


def roll_n(X, axis, n):
    f_idx = tuple(slice(None, None, None) if i != axis else slice(0, n, None) for i in range(X.dim()))
    b_idx = tuple(slice(None, None, None) if i != axis else slice(n, None, None) for i in range(X.dim()))
    front = X[f_idx]
    back = X[b_idx]
    return torch.cat([back, front], axis)


    # a: template; b: source
    # imshow(a.squeeze(0).float())
    # [B, 1, cfg.num_ring, cfg.num_sector, 2]
    eps = 1e-15

    real_a = torch.from_numpy(a[...,0]).to(device)
    real_b = torch.from_numpy(b[...,0]).to(device)
    imag_a = torch.from_numpy(a[...,1]).to(device)
    imag_b = torch.from_numpy(b[...,1]).to(device)

    # compute a * b.conjugate; shape=[B,H,W,C]
    R = torch.FloatTensor(1, 1, cfg.num_ring, cfg.num_sector, 2).to(device)
    R[...,0] = real_a * real_b + imag_a * imag_b
    R[...,1] = real_a * imag_b - real_b * imag_a

    r0 = torch.sqrt(real_a ** 2 + imag_a ** 2 + eps) * torch.sqrt(real_b ** 2 + imag_b ** 2 + eps).to(device)
    R[...,0] = R[...,0].clone()/(r0 + eps).to(device)
    R[...,1] = R[...,1].clone()/(r0 + eps).to(device)

    corr = torch.ifft(R, 2)
    corr_real = corr[...,0]
    corr_imag = corr[...,1]
    corr = torch.sqrt(corr_real ** 2 + corr_imag ** 2 + eps)
    corr = fftshift2d(corr)

    corr = corr.squeeze(1)
    corr_wb = corr2soft(corr)
    corr_ang = torch.sum(corr_wb, 1, keepdim=False)

    angle = torch.argmax(corr)
    angle = angle % cfg.num_sector

    return angle, corr
    
def phase_corr(a, b, device, corr2soft):
    # a: template; b: source
    # imshow(a.squeeze(0).float())
    # [B, 1, cfg.num_ring, cfg.num_sector, 2]
    eps = 1e-15

    # real_a = torch.from_numpy(a[...,0]).to(device)
    # real_b = torch.from_numpy(b[...,0]).to(device)
    # imag_a = torch.from_numpy(a[...,1]).to(device)
    # imag_b = torch.from_numpy(b[...,1]).to(device)

    real_a = a[...,0]
    real_b = b[...,0]
    imag_a = a[...,1]
    imag_b = b[...,1]

    # compute a * b.conjugate; shape=[B,H,W,C]
    R = torch.FloatTensor(1, 1, cfg.num_ring, cfg.num_sector, 2).to(device)
    R[...,0] = real_a * real_b + imag_a * imag_b
    R[...,1] = real_a * imag_b - real_b * imag_a

    r0 = torch.sqrt(real_a ** 2 + imag_a ** 2 + eps) * torch.sqrt(real_b ** 2 + imag_b ** 2 + eps).to(device)
    R[...,0] = R[...,0].clone()/(r0 + eps).to(device)
    R[...,1] = R[...,1].clone()/(r0 + eps).to(device)

    #R = a*b.conj()
    corr = torch.ifft(R, 2)
    corr_real = corr[...,0]
    corr_imag = corr[...,1]
    corr = torch.sqrt(corr_real ** 2 + corr_imag ** 2 + eps)
    corr = fftshift2d(corr)

    corr = corr.squeeze(1)
    corr_wb = corr2soft(corr)
    corr_ang = torch.sum(corr_wb, 1, keepdim=False)

    angle = torch.argmax(corr)
    angle = angle % cfg.num_sector

    return angle.detach().cpu().numpy(), corr


def phase_corr_MR(a, b):
    # a: template; b: source
    eps = 1e-15
    # temp = a*b.conj()
    # corr = np.fft.ifft2(temp.cpu().numpy(), norm="ortho")
    # corr = torch.from_numpy(corr).cuda()
    # corr = torch.sqrt(corr.imag**2 + corr.real**2 + eps)

    corr = torch.ifft(a*b.conj(),signal_ndim = 2 , normalized=True)
    # corr = torch.ifft(torch.ifft(a*b.conj(),signal_ndim = 1 , normalized=True) , signal_ndim = 1 , normalized=True)
    #print(corr)
    x_real = corr[...,0]
    #print('x_real：',x_real.size())
    x_imag = corr[...,1]
    #print('x_imag：',x_imag.size())
    #print(x_real)
    #print(x_imag)
    corr = torch.sqrt(x_imag**2 + x_real**2 + eps)

    corr = fftshift2d(corr)

    corr = corr.squeeze(1)

    angle = torch.argmax(corr)
    angle = angle % cfg.num_sector

    return angle.detach().cpu().numpy(), corr
