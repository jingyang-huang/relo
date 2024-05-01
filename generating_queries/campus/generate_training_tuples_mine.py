import os
import pickle
import random
import time
import numpy as np
import pandas as pd
from sklearn.neighbors import KDTree
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import math
import struct
import config as cfg
import gputransform
import linecache
import re
from scipy.spatial.transform import Rotation as R
import open3d as o3d

BASE_DIR = "/home/hjy/CampusData16/"
runs_folder = "/home/hjy/CampusData16/"
pointcloud_fols = "/Scans/"

testset_pick = 2
test_rate = 0.3
test_area = [125.0, 150.0, -50.0, 150.0] #test_area是一个范围，表示位置是否在范围内，在范围内的用于测试.可以提高模型的泛化能力

# check if the location is in the test set
def check_in_test_set(northing, easting, points):
    in_test_set = False
    if(points[0] < northing and northing < points[1] and points[2] < easting and easting < points[3]):
        in_test_set = True
    return in_test_set


# check if it's a new place in test set
def check_submap_test(northing, easting, prev_northing, prev_easting):
    is_submap = False
    euclidean = np.abs(np.sqrt((prev_northing-northing)**2 + (prev_easting-easting)**2))
    if(euclidean < cfg.SUBMAP_INTERVAL_TEST + 2.0 and euclidean >= cfg.SUBMAP_INTERVAL_TEST): #0.5
        is_submap = True
    return is_submap

#检查距离是否超过0.5 or 3m，可以保证数据集里的数据没有重复的
# check if it's a new place in train set
def check_submap_train(northing, easting, prev_northing, prev_easting):
    is_submap = False
    euclidean = np.abs(np.sqrt((prev_northing-northing)**2 + (prev_easting-easting)**2))
    if(euclidean < cfg.SUBMAP_INTERVAL_TRAIN + 2.0 and euclidean >= cfg.SUBMAP_INTERVAL_TRAIN): #1.0
        is_submap = True
    return is_submap


# # find closest place timestamp with index returned
# # 返回最近的tf的pcd
# def find_closest_timestamp(A, target):
#     # A must be sorted
#     idx = A.searchsorted(target)
#     idx = np.clip(idx, 1, len(A)-1)
#     left = A[idx-1]
#     right = A[idx]
#     idx -= target - left < right - target
#     return idx


def get_line_context(file_path, line_number):
    return linecache.getline(file_path, line_number).strip()

def getTransfromPose(pose):
    #print(pose)
    #tlist = [pose[3],pose[7],pose[11]]
    trans = [pose[3],pose[7],pose[11]]
    #print(trans)
    return trans

def getYawfromPose(pose):
    pitch, roll, yaw = R.from_matrix(pose).as_euler('xyz')
    #print(pose)
    #z = math.atan2(pose[1,0], pose[0,0])
    # print(pitch, roll, yaw)
    return yaw

def getXYandYaw(text):
    text = re.sub('\s+', ' ',text)
    text = text.strip()
    text = text.replace(" ", ",")
    str_list = eval(text)
    pose_SE3=np.array(str_list)
    trans = getTransfromPose(pose_SE3)
    pose_SE3 = np.reshape(pose_SE3, (-1, 4))
    # print(pose_SE3)
    pose_SE3 = pose_SE3[0:3,0:3]
    # print(pose_SE3)
    R = pose_SE3
    yaw = getYawfromPose(R)
    return trans[0],trans[1],yaw

# 不需要
# # nclt pointcloud utils
# def convert(x_s, y_s, z_s):
#     scaling = 0.005 # 5 mm
#     offset = -100.0

#     x = x_s * scaling + offset
#     y = y_s * scaling + offset
#     z = z_s * scaling + offset

#     return x, y, z
    

def load_pc_file(filename):
    pcd = o3d.io.read_point_cloud(os.path.join(BASE_DIR, filename))
    # print(np.asarray(pcd.points))
    hits = []
    points = np.asarray(pcd.points)
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

# # 重写成我的函数
# # load lidar file in nclt dataset
# def load_lidar_file_nclt(file_path):
#     n_vec = 4
#     f_bin = open(file_path,'rb')
#     #print("opening")

#     hits = []

#     while True:

#         x_str = f_bin.read(2)
#         if x_str == b"": # eof
#             break

#         x = struct.unpack('<H', x_str)[0]
#         y = struct.unpack('<H', f_bin.read(2))[0]
#         z = struct.unpack('<H', f_bin.read(2))[0]
#         i = struct.unpack('B', f_bin.read(1))[0]
#         l = struct.unpack('B', f_bin.read(1))[0]

#         x, y, z = convert(x, y, z)
#         s = "%5.3f, %5.3f, %5.3f, %d, %d" % (x, y, z, i, l)

#         # filter and normalize the point cloud to -1 ~ 1
#         if np.abs(x) < 70. and z > -20. and z < -2. and np.abs(y) < 70. and not(np.abs(x) < 1. and np.abs(y) < 1.):
#             hits += [[x/70., y/70., z/20.]]

#     f_bin.close()
#     hits = np.asarray(hits)
#     hits[:, 2] = -hits[:, 2]

#     return hits


# # load pointcloud and process it using CUDA accelerate 
# def load_pc_file(filename):
#     # returns Nx3 matrix
#     # scale the original pointcloud 
#     #print("pc path",os.path.join(BASE_DIR, filename))
#     pc = load_lidar_file_nclt(os.path.join(BASE_DIR, filename))
    
#     # pc[:,0] = pc[:,0] / np.max(pc[:,0] + 1e-15) - 0.0001
#     # pc[:,1] = pc[:,1] / np.max(pc[:,1] + 1e-15) - 0.0001
#     # pc[:,2] = pc[:,2] / np.max(pc[:,2] + 1e-15) - 0.0001

#     # !Debug
#     # x = pc[...,0]
#     # y = pc[...,1]
#     # z = pc[...,2]
#     # fig2 = plt.figure()
#     # ax2 = Axes3D(fig2)
#     # ax2.scatter(x, y, z)
#     # plt.show()

#     size = pc.shape[0]
#     pc_img = np.zeros([cfg.num_height * cfg.num_ring * cfg.num_sector])
#     pc = pc.transpose().flatten().astype(np.float32)

#     transer = gputransform.GPUTransformer(pc, size, cfg.max_length, cfg.max_height, cfg.num_ring, cfg.num_sector, cfg.num_height, 1)
#     transer.transform()
#     point_t = transer.retreive()
#     point_t = point_t.reshape(-1, 3)
#     point_t = point_t[...,2]
#     pc_img = point_t.reshape(cfg.num_height, cfg.num_ring, cfg.num_sector)

#     pc = np.sum(pc_img, axis=0)
#     # plt.imshow(pc)
#     # plt.show()
#     return pc_img


def output_write_to_file(output, filename):
    filename = filename+'.txt'
    with open(filename, 'w') as file:
        file.write(str(output))
    print("Write Done ", filename)


# dump the tuples to pickle files for training
def output_to_file(output, filename):
    with open(filename, 'wb') as handle:
        pickle.dump(output, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print("Done ", filename)


# positive和negative?
# construct query dict for training
def construct_query_dict(df_centroids, filename, pickle_flag):
    tree = KDTree(df_centroids[['northing','easting']])
    
    # get neighbors pair
    ind_nn = tree.query_radius(df_centroids[['northing','easting']],r=2)
    
    # get far away pairs
    ind_r = tree.query_radius(df_centroids[['northing','easting']], r=3)
    queries = {}
    print("ind_nn",len(ind_nn))
    print("ind_r",len(ind_r))

    for i in range(len(ind_nn)):
        print("index",i,' / ',len(ind_nn))
        
        # get query info
        query = df_centroids.iloc[i]["file"]

        # get yaw info of this query
        query_yaw = df_centroids.iloc[i]["yaw"]

        # get positive filename and shuffle
        # 在此为最近的邻居集合除去i本身后留下的
        positives = np.setdiff1d(ind_nn[i],[i]).tolist()
        random.shuffle(positives)
        # positives = positives[0:2]

        # get negative filename and shuffle
        # 找到2个数组中集合元素的差异, 即求余下的集合，在此为除去ind_r后原数据集留下的所有
        negatives = np.setdiff1d(
            df_centroids.index.values.tolist(),ind_r[i]).tolist()
        random.shuffle(negatives)
        # negatives = negatives[0:50]
        
        # add all info to query dict
        queries[i] = {"query":query, "heading":query_yaw,
                      "positives":positives,"negatives":negatives}

    # dump all queries into pickle file for training
    if pickle_flag:
        output_to_file(queries,filename)
        output_write_to_file(queries,filename)
        # with open(filename, 'w') as file:
        #     file.write(str(queries))
        # print("Write Done ", filename)


if __name__ == "__main__":

    all_folders = sorted(os.listdir(BASE_DIR))
    
    folders = []
    velo_file = []

    # All runs are used for training (both full and partial)
    index_list = range(len(all_folders)) #-1会导致少一个
    print("Number of runs: " + str(len(index_list)))

    for index in index_list:
        # if index == 0:
        folders.append(all_folders[index])
    print(folders)

    for folder in folders:
        print(folder)

        velo_file = []

        if folder == folders[0]:
            save_flag = True #第一个folder为训练集
        else:
            save_flag = False

        # Initialize pandas DataFrame
        df_train = pd.DataFrame(columns=['file','northing','easting','yaw'])
        df_test = pd.DataFrame(columns=['file','northing','easting','yaw'])
        df_all = pd.DataFrame(columns=['file','northing','easting','yaw'])
        df_velo = pd.DataFrame(columns=['file','northing','easting','yaw'])

        # get groundtruth file and load it
        gt_filename = "tf/tf.txt"

        #这个载入需要自己写，因为格式是.txt
        location_array = []
        gtfile = open(os.path.join(BASE_DIR,folder,gt_filename)) 
        print(gtfile)
        line = gtfile.readline()               # 调用文件的 readline()方法 
        while line: 
            #print(line, end = '')     # 在 Python 3 中使用 
            x,y,yaw = getXYandYaw(line)
            location_array.append([x,y,yaw])
            line = gtfile.readline() 
        gtfile.close() 
        print('len',len(location_array))
        
        #print(Trans_data)
        
        df_locations = pd.DataFrame(location_array)
        df_locations.columns=['northing','easting','yaw']
        #print(df_locations)
        pcd_list= os.listdir(os.path.join(BASE_DIR, folder + pointcloud_fols))
        
        # get the file name
        for pcd_id in range(len(pcd_list)):
            pcnames = str(pcd_id)
            velo_file.append(pcnames)

        # convert data type
        df_all['file'] = velo_file

        # save all relative info into df_all
        # for idx in range(len(df_all)):
        #     loc_idx = idx #find_closest_timestamp(df_locations['timestamp'].values, int(df_all['file'][idx]))
        #     df_all['yaw'][idx] = df_locations['yaw'][loc_idx]
        #     df_all['northing'][idx] = df_locations['northing'][loc_idx]
        #     df_all['easting'][idx] = df_locations['easting'][loc_idx]

        df_all['yaw'] = df_locations['yaw']
        df_all['northing'] = df_locations['northing']
        df_all['easting'] = df_locations['easting']


        # get full path of the point cloud files
        df_all['file'] = BASE_DIR+folder + pointcloud_fols+df_all['file'].astype(str)+'.pcd' #这里给出了pcd的路径
        df_all.to_csv('./dfall.csv')
        # x = df_all['northing']
        # y = df_all['easting']
        # plt.scatter(x,y,s=1.0)
        # plt.show()

        first_flag = False
        #input()

        for index, row in df_all.iterrows():
            # print(row['file']) #在这里打印的file路径
            print("index", index, ' / ', len(df_all))
            # for not nan value and very first ones (which often wrong)
            if np.isnan(float(row['northing'])) or np.isnan(float(row['easting'])):
                continue
            elif not first_flag :
                #print('first_flag false')
                prev_northing, prev_easting = float(row['northing']), float(row['easting'])
                first_flag = True          

            if save_flag:
                if(check_submap_train(float(row['northing']), float(row['easting']), float(prev_northing), float(prev_easting))):
                    # print("\033[0;0;34m","save this submap","\033[0m")
                    # process point cloud and save
                    velo = load_pc_file(row['file']) #相当与提前把数据预处理好
                    save_name = row['file'].replace('.pcd','.npy')
                    row['file'] = row['file'].replace('.pcd','.npy')
                    save_name = save_name.replace(pointcloud_fols, cfg.TRAIN_FOLDER)
                    row['file'] = row['file'].replace(pointcloud_fols, cfg.TRAIN_FOLDER)
                    np.save(save_name, velo)

                    if(testset_pick == 1):
                        if(check_in_test_set(float(row['northing']), float(row['easting']), test_area)):
                            df_test = df_test.append(row, ignore_index=True)
                        else:
                            df_train = df_train.append(row, ignore_index=True)
                    elif(testset_pick == 2):
                        r = random.random()
                        if(r < test_rate):
                            df_test = df_test.append(row, ignore_index=True)
                        else:
                            df_train = df_train.append(row, ignore_index=True)
                    prev_northing, prev_easting = float(row['northing']), float(row['easting'])

            else:
                if(check_submap_test(float(row['northing']), float(row['easting']), float(prev_northing), float(prev_easting))):
                    # process point cloud and save
                    # print("\033[0;0;35m","save this submap","\033[0m")
                    velo = load_pc_file(row['file'])
                    save_name = row['file'].replace('.pcd','.npy')
                    row['file'] = row['file'].replace('.pcd','.npy')
                    save_name = save_name.replace(pointcloud_fols, cfg.TEST_FOLDER)
                    row['file'] = row['file'].replace(pointcloud_fols, cfg.TEST_FOLDER)
                    np.save(save_name, velo)
                    
                    if(testset_pick == 1):
                        if(check_in_test_set(float(row['northing']), float(row['easting']), test_area)):
                            df_test = df_test.append(row, ignore_index=True)
                        else:
                            df_train = df_train.append(row, ignore_index=True)
                    elif(testset_pick == 2):
                        r = random.random()
                        if(r < test_rate):
                            df_test = df_test.append(row, ignore_index=True)
                        else:
                            df_train = df_train.append(row, ignore_index=True)

                    prev_northing, prev_easting = float(row['northing']), float(row['easting'])

        if save_flag:
            print("Number of training submaps: "+str(len(df_train['file'])))
            print("Number of non-disjoint test submaps: "+str(len(df_test['file'])))
            construct_query_dict(df_train, "./training_queries_baseline_" + cfg.EXPERIMENT_NAME + ".pickle", pickle_flag=save_flag)
            construct_query_dict(df_test, "./test_queries_baseline_" + cfg.EXPERIMENT_NAME + ".pickle", pickle_flag=save_flag)

            gt_train_filename = "gt_" + cfg.EXPERIMENT_NAME + "_0.5m.csv"
            df_train.to_csv(os.path.join(BASE_DIR,folder,gt_train_filename))
            gt_test_filename = "gt_" + cfg.EXPERIMENT_NAME + "_test_0.5m.csv"
            df_test.to_csv(os.path.join(BASE_DIR,folder,gt_test_filename))
        
        else:          
            gt_train_filename = "gt_" + cfg.EXPERIMENT_NAME + "_3m.csv"
            df_train.to_csv(os.path.join(BASE_DIR,folder,gt_train_filename))
            gt_test_filename = "gt_" + cfg.EXPERIMENT_NAME + "_test_3m.csv"
            df_test.to_csv(os.path.join(BASE_DIR,folder,gt_test_filename))
