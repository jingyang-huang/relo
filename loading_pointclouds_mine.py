import os
import pickle
import numpy as np
import random
import config as cfg
import struct
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import gputransform
import time
import faulthandler
faulthandler.enable()

def get_queries_dict(filename):
    # key:{'query':file,'positives':[files],'negatives:[files], 'neighbors':[keys]}
    with open(filename, 'rb') as handle:
        queries = pickle.load(handle)
        print("Queries Loaded.")
        return queries


def get_sets_dict(filename):
    #[key_dataset:{key_pointcloud:{'query':file,'northing':value,'easting':value}},key_dataset:{key_pointcloud:{'query':file,'northing':value,'easting':value}}, ...}
    with open(filename, 'rb') as handle:
        trajectories = pickle.load(handle)
        print("Trajectories Loaded.")
        return trajectories



def load_array_infer(points):
    hits = []
    points = points[:, 0:3]
    #print("1-0")

    #这样既快而且不会卡死
    #hits = np.asarray(points)

    pc = np.asarray(points)
    #hits的预处理，即提前滤波是必须的
    #还加入了滤除地面
    #print(pc.shape)
    hits = pc[np.where((np.abs(pc[...,0]) < 100.)&(np.abs(pc[...,1]) < 100.)&(np.abs(pc[...,2]) < 30.)  &(np.abs(pc[:,0]) > 1.)&(np.abs(pc[:,1]) > 1.)&(np.abs(pc[:,2]) > 1.0))]
    #hits = pc[np.where((np.abs(pc[:,0]) < 70.)&(np.abs(pc[:,1]) < 70.)&(np.abs(pc[:,2]) < 20.)&(np.abs(pc[:,2]) > 2.)&(np.abs(pc[:,0]) > 5.)&(np.abs(pc[:,1]) > 5.))]
    
    hits[...,0] = hits[...,0] / 100.
    hits[...,1] = hits[...,1] / 100.
    hits[...,2] = hits[...,2] / 30.
    #print("hits len is",len(hits))

    # for pt in points:
    #         #print(pt)
    #         x = pt[0]
    #         y = pt[1]
    #         z = pt[2]
    #         #x, y, z = convert(x, y, z)
    #         s = "%5.3f, %5.3f, %5.3f" % (x, y, z)
    #         print(s)
    #         # filter and normalize the point cloud to -1 ~ 1
    #         if np.abs(x) < 100. and np.abs(y) < 100. and z > -30. and z < 30.  :  #and not(np.abs(x) < 5. and np.abs(y) < 5.)
    #             hits += [[x/100., y/100., z/30.]]
    # hits = np.asarray(hits)
    # print("hits len is",len(hits))

    #or
    # vector = np.array([150.,200.,50.])
    # hits =  points / vector

    # 这一步可以用其他方法实现，直接放在segment里面
    '''
    for pt in points:
            #print(pt)
            x = pt[0]
            y = pt[1]
            z = pt[2]
            #x, y, z = convert(x, y, z)
            s = "%5.3f, %5.3f, %5.3f" % (x, y, z)
            #print(s)
            # filter and normalize the point cloud to -1 ~ 1
            #if np.abs(x) < 70. and z > -20. and z < -2. and np.abs(y) < 70. and not(np.abs(x) < 5. and np.abs(y) < 5.):
            hits += [[x/150., y/200., z/50.]]

    hits = np.asarray(hits)
    '''

    #hits[:, 2] = -hits[:, 2]
    #print(hits)

    #pc = np.array(hits, dtype=np.float32)
    pc = hits
    size = pc.shape[0]
    #print("1-1")
    pc_img = np.zeros([cfg.num_height * cfg.num_ring * cfg.num_sector])
    pc = pc.transpose().flatten().astype(np.float32)
    #print("1-2")

    transer = gputransform.GPUTransformer(pc, size, cfg.max_length, cfg.max_height, cfg.num_ring, cfg.num_sector, cfg.num_height, 1)
    transer.transform()
    #print("1-3")
    point_t = transer.retreive()
    #print("1-4")
    point_t = point_t.reshape(-1, 3)
    point_t = point_t[...,2]
    pc_img = point_t.reshape(cfg.num_height, cfg.num_ring, cfg.num_sector)
    #print("1-5")
    return pc_img


def load_pc_file(filename):
    filename = filename.replace('.bin','.npy')
    filename = filename.replace('/velo_trans/','/occ_0.5m/')
    pc_img = np.load(filename)

    return pc_img


def load_pc_files(filenames):
    pcs = []
    for filename in filenames:
        pc = load_pc_file(filename)
        pcs.append(pc)
    pcs = np.array(pcs)
    return pcs


def rotate_point_cloud(batch_data):
    """ Randomly rotate the point clouds to augument the dataset
        rotation is per shape based along up direction
        Input:
          BxNx3 array, original batch of point clouds
        Return:
          BxNx3 array, rotated batch of point clouds
    """
    rotated_data = np.zeros(batch_data.shape, dtype=np.float32)
    for k in range(batch_data.shape[0]):
        #rotation_angle = np.random.uniform() * 2 * np.pi
        #-90 to 90
        rotation_angle = (np.random.uniform()*np.pi) - np.pi/2.0
        cosval = np.cos(rotation_angle)
        sinval = np.sin(rotation_angle)
        rotation_matrix = np.array([[cosval, -sinval, 0],
                                    [sinval, cosval, 0],
                                    [0, 0, 1]])
        shape_pc = batch_data[k, ...]
        rotated_data[k, ...] = np.dot(
            shape_pc.reshape((-1, 3)), rotation_matrix)
    return rotated_data


def jitter_point_cloud(batch_data, sigma=0.005, clip=0.05):
    """ Randomly jitter points. jittering is per point.
        Input:
          BxNx3 array, original batch of point clouds
        Return:
          BxNx3 array, jittered batch of point clouds
    """
    B, N, C = batch_data.shape
    assert(clip > 0)
    jittered_data = np.clip(sigma * np.random.randn(B, N, C), -1*clip, clip)
    jittered_data += batch_data
    return jittered_data

#应该没问题，但是一会pos一会neg挺乱的
def get_query_tuple(dict_value, num_pos, num_neg, QUERY_DICT, hard_neg=[], other_neg=False):
    # get query tuple for dictionary entry
    # return list [query,positives,negatives]
    heading = []
    query = load_pc_file(dict_value["query"])  # Nx3
    heading.append(dict_value["heading"])

    random.shuffle(dict_value["positives"])
    pos_files = []
    
    for i in range(num_pos):
        pos_files.append(QUERY_DICT[dict_value["positives"][i]]["query"])
        heading.append(QUERY_DICT[dict_value["positives"][i]]["heading"])
    positives = load_pc_files(pos_files)

    neg_files = []
    neg_indices = []
    if(len(hard_neg) == 0):
        random.shuffle(dict_value["negatives"])
        for i in range(num_neg):
            neg_files.append(QUERY_DICT[dict_value["negatives"][i]]["query"])
            neg_indices.append(dict_value["negatives"][i])
            heading.append(QUERY_DICT[dict_value["negatives"][i]]["heading"])

    else:
        random.shuffle(dict_value["negatives"])
        for i in hard_neg:
            neg_files.append(QUERY_DICT[i]["query"])
            heading.append(QUERY_DICT[i]["heading"])

            neg_indices.append(i)
        j = 0
        while(len(neg_files) < num_neg):

            if not dict_value["negatives"][j] in hard_neg:
                neg_files.append(
                    QUERY_DICT[dict_value["negatives"][j]]["query"])
                heading.append(QUERY_DICT[dict_value["negatives"][j]]["heading"])

                neg_indices.append(dict_value["negatives"][j])
            j += 1

    negatives = load_pc_files(neg_files)

    if other_neg is False:
        return [query, positives, negatives, heading]
    # For Quadruplet Loss
    else: #一般是true ，  但是这里面的东西好乱啊
        # get neighbors of negatives and query
        neighbors = []
        for pos in dict_value["positives"]:
            neighbors.append(pos)
        for neg in neg_indices:
            for pos in QUERY_DICT[neg]["positives"]:
                neighbors.append(pos)
        possible_negs = list(set(QUERY_DICT.keys())-set(neighbors))
        random.shuffle(possible_negs)

        if(len(possible_negs) == 0):
            return [query, positives, negatives, np.array([]), heading]

        neg2 = load_pc_file(QUERY_DICT[possible_negs[0]]["query"])
        heading.append(QUERY_DICT[possible_negs[0]]["heading"])
        heading = np.array(heading)

        return [query, positives, negatives, neg2, heading]


def get_rotated_tuple(dict_value, num_pos, num_neg, QUERY_DICT, hard_neg=[], other_neg=False):
    query = load_pc_file(dict_value["query"])  # Nx3
    q_rot = rotate_point_cloud(np.expand_dims(query, axis=0))
    q_rot = np.squeeze(q_rot)

    random.shuffle(dict_value["positives"])
    pos_files = []
    for i in range(num_pos):
        pos_files.append(QUERY_DICT[dict_value["positives"][i]]["query"])
    #positives= load_pc_files(dict_value["positives"][0:num_pos])
    positives = load_pc_files(pos_files)
    p_rot = rotate_point_cloud(positives)

    neg_files = []
    neg_indices = []
    if(len(hard_neg) == 0):
        random.shuffle(dict_value["negatives"])
        for i in range(num_neg):
            neg_files.append(QUERY_DICT[dict_value["negatives"][i]]["query"])
            neg_indices.append(dict_value["negatives"][i])
    else:
        random.shuffle(dict_value["negatives"])
        for i in hard_neg:
            neg_files.append(QUERY_DICT[i]["query"])
            neg_indices.append(i)
        j = 0
        while(len(neg_files) < num_neg):
            if not dict_value["negatives"][j] in hard_neg:
                neg_files.append(
                    QUERY_DICT[dict_value["negatives"][j]]["query"])
                neg_indices.append(dict_value["negatives"][j])
            j += 1
    negatives = load_pc_files(neg_files)
    n_rot = rotate_point_cloud(negatives)

    if other_neg is False:
        return [q_rot, p_rot, n_rot]

    # For Quadruplet Loss
    else:
        # get neighbors of negatives and query
        neighbors = []
        for pos in dict_value["positives"]:
            neighbors.append(pos)
        for neg in neg_indices:
            for pos in QUERY_DICT[neg]["positives"]:
                neighbors.append(pos)
        possible_negs = list(set(QUERY_DICT.keys())-set(neighbors))
        random.shuffle(possible_negs)

        if(len(possible_negs) == 0):
            return [q_jit, p_jit, n_jit, np.array([])]

        neg2 = load_pc_file(QUERY_DICT[possible_negs[0]]["query"])
        n2_rot = rotate_point_cloud(np.expand_dims(neg2, axis=0))
        n2_rot = np.squeeze(n2_rot)

        return [q_rot, p_rot, n_rot, n2_rot]


def get_jittered_tuple(dict_value, num_pos, num_neg, QUERY_DICT, hard_neg=[], other_neg=False):
    query = load_pc_file(dict_value["query"])  # Nx3
    q_jit = jitter_point_cloud(np.expand_dims(query, axis=0))
    q_jit = np.squeeze(q_jit)

    random.shuffle(dict_value["positives"])
    pos_files = []
    for i in range(num_pos):
        pos_files.append(QUERY_DICT[dict_value["positives"][i]]["query"])
    positives = load_pc_files(pos_files)
    p_jit = jitter_point_cloud(positives)

    neg_files = []
    neg_indices = []
    if(len(hard_neg) == 0):
        random.shuffle(dict_value["negatives"])
        for i in range(num_neg):
            neg_files.append(QUERY_DICT[dict_value["negatives"][i]]["query"])
            neg_indices.append(dict_value["negatives"][i])
    else:
        random.shuffle(dict_value["negatives"])
        for i in hard_neg:
            neg_files.append(QUERY_DICT[i]["query"])
            neg_indices.append(i)
        j = 0
        while(len(neg_files) < num_neg):
            if not dict_value["negatives"][j] in hard_neg:
                neg_files.append(
                    QUERY_DICT[dict_value["negatives"][j]]["query"])
                neg_indices.append(dict_value["negatives"][j])
            j += 1
    negatives = load_pc_files(neg_files)
    n_jit = jitter_point_cloud(negatives)

    if other_neg is False:
        return [q_jit, p_jit, n_jit]

    # For Quadruplet Loss
    else:
        # get neighbors of negatives and query
        neighbors = []
        for pos in dict_value["positives"]:
            neighbors.append(pos)
        for neg in neg_indices:
            for pos in QUERY_DICT[neg]["positives"]:
                neighbors.append(pos)
        possible_negs = list(set(QUERY_DICT.keys())-set(neighbors))
        random.shuffle(possible_negs)

        if(len(possible_negs) == 0):
            return [q_jit, p_jit, n_jit, np.array([])]

        neg2 = load_pc_file(QUERY_DICT[possible_negs[0]]["query"])
        n2_jit = jitter_point_cloud(np.expand_dims(neg2, axis=0))
        n2_jit = np.squeeze(n2_jit)

        return [q_jit, p_jit, n_jit, n2_jit]
