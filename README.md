# 基于DiSCO-Pytorch的重定位功能

## 环境配置与Disco的安装/部署
对X86的Ubuntu系统，首先应保证系统安装了ROS和PCL库。

然后需要安装python环境，一般建议使用conda建立虚拟环境。进入镜像站`https://repo.anaconda.com/archive/`，下载对应的版本。(X20一般是x86)
创建环境`conda create --name py_relo python=3.8`

之后进入虚拟环境`conda activate py_relo`，执行:

1.pip加速 `pip config set global.index-url https://pypi.mirrors.ustc.edu.cn/simple/ `

2.`pip install -r requirements_noros.txt`  
因为南瑞的狗没法联网，所以也可以把配置好的conda环境直接拷贝过去

3.安装cuda(对于有GPU的环境): `sudo apt install nvidia-cuda-toolkit` ,经测试好像不用安装cudnn也能运行，而且似乎这样安装的cuda版本很低，不过能运行也没深究。

这样就安装完了Disco的依赖:

>**Pre-Requisites**
>* PyTorch 1.4.0 (<= 1.6.0 fft module is modified after 1.6.0)
>* tensorboardX
>* Cython (for point cloud process [voxelization](https://github.com/ZJU-Robotics-Lab/Voxelization_API.git))

 
4.安装gputransform

对于有gpu的环境，安装cuda后，通过这个.so可以把点云载入显卡加速点云预处理的过程；而对于没有gpu的环境，为了兼容性，选择`multi-layer-polar-cython-cpu`，也执行一样的步骤获得.so。

此处引用Disco的部署方法(gpu版)

>**Use point cloud process module in cuda (cython wrapped)**

>In [multi-layer-polar-cython-gpu](https://github.com/MaverickPeter/DiSCO-pytorch/tree/main/multi-layer-polar-cython)/[cython](https://github.com/MaverickPeter/DiSCO-pytorch/tree/main/multi-layer-polar-cython/cython)

```
# To inplace install the cython wrapped module:
python setup.py build_ext --inplace

# to test
python test.py 

(需要注意，如果遇到了segmentation fault error，说明点过多了，需要在当前的终端输入'ulimit -s 81920'来提高系统的堆大小)
(If you meet segmentation fault error, you may have overlarge number of points to process e.g. 67w. To tackle this problem you may need to change your system stack size by 'ulimit -s 81920' in your bash)
```

>and now you will have a gputransform.cpythonxxx.so file and copy it to **[generating_queries](https://github.com/MaverickPeter/DiSCO-pytorch/tree/main/generating_queries)/[nclt](https://github.com/MaverickPeter/DiSCO-pytorch/tree/main/generating_queries/nclt) and main dir** where you can find a place holder. Note that the input of the wrapped point cloud process module you should scale the original point cloud to **[-1~1]** range for all axis. No extra process for point cloud such as downscale the points number.

>之后会在目录下得到gputransform.cpythonxxx.so文件，然后把它复制到generating_queries/nclt目录下以及主目录下(和本Readme同目录)。


## 使用
### 训练
为了保证重定位的精度，需要对当前的巡检路径进行训练，这部分比较麻烦：

1.需要使用**机器狗**录制4个数据包，三个用于训练，一个用于提供重定位参考，命令`rosbag record -O data-4.bag /velodyne_points /imu/data /tf_static /leg_odom`。

ps.感觉如果录数据集(训练和参考都是)的时候，跑慢一点<1m/s，那么重定位效果会更好

2.录制完数据包后，对每个包，载入同一张地图，使用**电脑**跑一遍定位(也可以用狗跑，不过狗跑的话，算力有限，发布的tf和点云很多对不齐，会缺少很多帧)。可以录制包`rosbag record -O data-1-tf.bag /velodyne_points /tf_static /tf`，也可以同时运行第3步来获取帧。

3.使用**电脑**播放包or运行2的同时，运行本工作空间下的save_frames包,`roslaunch save_frame save_frame.launch`，可以指定保存的位置。
指定保存位置的命令为`roslaunch save_frame save_frame.launch RecordPrefix:=/home/hjy/testData/1/Scans/ GroundTruthPrefix:=/home/hjy/testData/1/tf/`

4.按照和Disco类似的方式组织训练数据，即:
```
└──CampusData
    ├── 1
    │   ├── Scans
    │   ├── occ_0.5m
    │   └── tf
    ├── 2
    │   ├── Scans
    │   ├── occ_3m
    │   └── tf
    ├── 3
    │   ├── Scans
    │   ├── occ_3m
    │   └── tf
...
(至少需要3组)
```
其中 **(occ_xm 是空文件夹)**

进入 [generating_queries](https://github.com/MaverickPeter/DiSCO-pytorch/tree/main/generating_queries)/[nclt](https://github.com/MaverickPeter/DiSCO-pytorch/tree/main/generating_queries/nclt)/
，依次执行下面两个文件
```
python generate_training_tuples_baseline_with_pose.py
python generate_test_sets.py
```
生成完了之后，可以在CampusData/1/下看到gt_occ_0.5m.csv和gt_occ_test_0.5m.csv，在generating_queries/nclt/下，看到dfall.csv、evaluation_database.pickle等很多文件。

然后执行开始训练，一般不需要再指定参数
```
python train_DiSCO.py (arguments please refer to the code in this python file)
```
训练完成后，可以在/log/下看到model.ckpt，后续重定位的推理会用到这个模型。把这个模型复制到要部署的环境的/log/下。

### 建立参考关键帧及轨迹
编译完成后需要建立参考关键帧及轨迹，进入relocalization的python环境，执行mapseg_from_liosam.py，会在指定目录下生成每个scan的.npy。

### 重定位包的编译和接口
目前重定位模块有两种接口，第一种写成了ros service的形式，第二种写成了节点形式。

1.service形式的输入和输出都为geometry_msgs/Pose2D，如果需要使用client调用这个服务，那么要在自己的包里复制/srv/relocalize_pointcloud.srv，在package.xml添加：
```
<build_depend>message_generation</build_depend>
<run_depend>message_runtime</run_depend>
```
并且在CMakeLists.txt中添加:
```
find_package(catkin REQUIRED COMPONENTS 
  roscpp 
  rospy 
  sensor_msgs
  geometry_msgs
  std_msgs
  nav_msgs
  message_generation  
 )

add_service_files(
  DIRECTORY srv
  FILES
  relocalize_pointcloud.srv
)

generate_messages(
  DEPENDENCIES
  nav_msgs
  std_msgs
  sensor_msgs
  geometry_msgs
)

catkin_package(
  CATKIN_DEPENDS    
  geometry_msgs
  sensor_msgs
  std_msgs  
  nav_msgs
  message_runtime
)
```
client的编写可以参考/scripts/client.py。
具体的调用方法是：

1.client接入服务节点，传入之前的定位(Pose2D)

2.server执行重定位，在传入定位的附近查找最可能的位置，并返回给client

3.本次service结束，client获得重定位结果

service部分的代码就在本目录下，选择工作空间后catkin_make即可。
不过由于安装了虚拟环境，可能编译会报错缺少python3-empy，这时候改为 
catkin_make -DPYTHON_EXECUTABLE=/usr/bin/python3即可

2.节点形式直接兼容了现在的定位，会订阅定位丢失的话题location_status，当检测到1时，执行一次重定位，并且把新的定位结果发布到/initialpose
测试方法rostopic pub  /location_status  std_msgs/Int8 1 


## 运行重定位

新开一个终端，source devel/setup.bash。

若要以service模式运行，则python relocalization_rossrv.py，获取重定位需要执行clinet.py；若要以节点模式运行，python relocalization_rosnode.py。


## 其余部分
参考README_Disco.md的说明
除了已经提到的mapseg_from_liosam.py、relocalization_rossrv.py等程序，还有一些用于测试的脚本文件：
- evaluate.py --用于测试模型训练质量
- loading_pointclouds_mine.py --加载点云
- lookup_liosam.py --测试单个pcd，返回最近几个的pcd
- lookup_liosam_batch.py --测试一个文件夹里的pcd，通过tf衡量准确程度
- lookup_liosam_batch_icp.py --测试一个文件夹里的pcd，通过tf衡量准确程度，并使用icp找出fitness最高的
- config.py里记录了许多参数，比如：
  - DATA_TYPE 这里因为是以园区作为数据集，所以名称是campus
  - SUBMAP_INTERVAL_TRAIN 和 SUBMAP_INTERVAL_TEST 用于分割提取出位置间隔大于INTERVAL的训练集和测试集
