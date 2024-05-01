#include <sys/stat.h>
#include <Eigen/Core>
#include <Eigen/Dense>

#include <vector>
#include <string>
#include <algorithm>
#include <filesystem>
#include <fstream>　// c++文件操作

#include <pcl/io/pcd_io.h>
#include <pcl/common/common_headers.h>
#include <pcl/segmentation/sac_segmentation.h> //基于采样一致性分割的类的头文件
#include <pcl/ModelCoefficients.h>
#include <pcl/io/boost.h>
#include <pcl/common/transforms.h>
#include <pcl/point_types.h>
#include <pcl/filters/passthrough.h>

#include <fstream>

//the following are UBUNTU/LINUX ONLY terminal color codes.
#define RESET   "\033[0m"
#define BLACK   "\033[30m"      /* Black */
#define RED     "\033[31m"      /* Red */
#define GREEN   "\033[32m"      /* Green */
#define YELLOW  "\033[33m"      /* Yellow */
#define BLUE    "\033[34m"      /* Blue */
#define MAGENTA "\033[35m"      /* Magenta */
#define CYAN    "\033[36m"      /* Cyan */
#define WHITE   "\033[37m"      /* White */
#define BOLDBLACK   "\033[1m\033[30m"      /* Bold Black */
#define BOLDRED     "\033[1m\033[31m"      /* Bold Red */
#define BOLDGREEN   "\033[1m\033[32m"      /* Bold Green */
#define BOLDYELLOW  "\033[1m\033[33m"      /* Bold Yellow */
#define BOLDBLUE    "\033[1m\033[34m"      /* Bold Blue */
#define BOLDMAGENTA "\033[1m\033[35m"      /* Bold Magenta */
#define BOLDCYAN    "\033[1m\033[36m"      /* Bold Cyan */
#define BOLDWHITE   "\033[1m\033[37m"      /* Bold White */

using namespace std;
static std::vector<std::string> file_lists;
std::string PcdSavePrefix = "/home/dr-slamintern-hjy/Disco_ws/src/Disco-test/tools/submaps_pcd_compressed/";
std::string BinSavePrefix = "/home/dr-slamintern-hjy/Disco_ws/src/Disco-test/tools/submaps_bin/";
std::string ReadMmapfile = "/home/dr-slamintern-hjy/Disco_ws/src/Disco-test/tools/campus_2.pcd";
std::string SaveFilename = "parameters.txt";
bool allow_plane_align = 0;
Eigen::Affine3f inner_transform;
ofstream outfile;

void read_filelists(const std::string& dir_path,std::vector<std::string>& out_filelsits,std::string type)
{
    struct dirent *ptr;
    DIR *dir;
    dir = opendir(dir_path.c_str());
    out_filelsits.clear();
    while ((ptr = readdir(dir)) != NULL){
        std::string tmp_file = ptr->d_name;
        if (tmp_file[0] == '.')continue;
        if (type.size() <= 0){
            out_filelsits.push_back(ptr->d_name);
        }else{
            if (tmp_file.size() < type.size())continue;
            std::string tmp_cut_type = tmp_file.substr(tmp_file.size() - type.size(),type.size());
            if (tmp_cut_type == type){
                out_filelsits.push_back(ptr->d_name);
            }
        }
    }
}

void convertPCDtoBin(std::string &in_file, std::string& out_file)
{
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);

    if (pcl::io::loadPCDFile<pcl::PointXYZ>(in_file, *cloud) == -1) //* load the file
    {
        std::string err = "Couldn't read file " + in_file;
        PCL_ERROR(err.c_str());
        return;// (-1);
    }
    // std::cout << "Loaded "
    //         << cloud->width * cloud->height
    //         << " data points from " 
    //         << in_file
    //         << " with the following fields: "
    //         << std::endl;

    int data_idx = 0;
    std::ostringstream oss;
    oss << pcl::PCDWriter::generateHeader(*cloud);// << "DATA binary\n";
    oss.flush();
    data_idx = static_cast<int>(oss.tellp());

    std::vector<pcl::PCLPointField> fields;
    std::vector<int> fields_sizes;
    size_t fsize = 0;
    size_t data_size = 0;
    size_t nri = 0;
    pcl::getFields (*cloud, fields);

    // Compute the total size of the fields
    for (const auto &field : fields)
    {
        if (field.name == "_")
        {
            continue;
        }
        
        int fs = field.count * pcl::getFieldSize (field.datatype);
        fsize += fs;
        fields_sizes.push_back (fs);
        fields[nri++] = field;
    }
    fields.resize (nri);

    data_size = cloud->points.size () * fsize;
    const int memsize = cloud->points.size() * sizeof(float) * 4;
    char *out = (char*)malloc( memsize);// 4 field x y z intensity
    //std::cout << "data_size size: " << data_size << std::endl;
    
    // char buffer[100];
    std::ofstream myFile (out_file.c_str(), std::ios::out | std::ios::binary);
    

    for (size_t i = 0; i < cloud->points.size(); ++i)
    {
        int nrj = 0;
        for (const auto &field : fields)
        {
            memcpy(out, reinterpret_cast<const char*> (&cloud->points[i]) + field.offset, fields_sizes[nrj++]);
            //myFile.write (reinterpret_cast<const char*> (&cloud->points[i]) + field.offset, fields_sizes[nrj++]);
        }
        float intensity = 0;
        memcpy(out, reinterpret_cast<const char*> (&intensity) , sizeof(intensity));
        // myFile.write ( reinterpret_cast<const char*> (&intensity) , sizeof(intensity));
    }
    myFile.write(out, memsize);

    myFile.close();
}

void PCLtoBin(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, std::string& out_file)
{
    pcl::PCDReader reader;	//定义点云读取对象
    pcl::PCDWriter writer;
    //writer.writeBinary(out_file, *cloud);
    writer.writeASCII(out_file, *cloud);

    // //int data_idx = 0;
    // std::ostringstream oss;
    // oss << pcl::PCDWriter::generateHeader(*cloud) << "DATA ascii\n";

    // //oss.flush();
    // //data_idx = static_cast<int>(oss.tellp());

    // const int memsize = cloud->points.size() * sizeof(float) * 5;
    // char *out = (char*)malloc( memsize);// 4 field x y z intensity

    // // char buffer[100];
    // std::ofstream myFile (out_file.c_str(), std::ios::out |std::ios::trunc); //| std::ios::binary
    // if(!myFile.good()) cout<<"Couldn't open "<<out_file<<endl;  
    // myFile << oss ;

    // //PCD 2 BIN 
    // for (size_t i = 0; i < cloud->points.size (); ++i)
    // {
    //     myFile.write((char*)&cloud->points[i].x,3*sizeof(float)); 
    //     //myFile.write((char*)&cloud->points[i].intensity,sizeof(float));
    // }

    // myFile.close();
}


int main(int argc, char **argv)
{
    std::cout << "Run map transform" << std::endl;
    outfile.open(PcdSavePrefix + SaveFilename , ios::trunc| ios::in | ios::out); //TODO:后期可以u换成binary
    //读取点云
    pcl::PointCloud<pcl::PointXYZ>::Ptr map_read(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::PointCloud<pcl::PointXYZ>::Ptr Mmap(new pcl::PointCloud<pcl::PointXYZ>);
    //打开点云文件
    if (pcl::io::loadPCDFile<pcl::PointXYZ>(ReadMmapfile, *map_read) == -1)//必须用绝对路径吗
    {
        string err_info = "Couldn't read file " + ReadMmapfile + "\n" ; 
        PCL_ERROR(err_info.data());
        //return ;
    }
    
    Eigen::Matrix4f align_rotation_inverse = Eigen::Matrix4f::Identity();
    //找到地面法向量，实现点云的水平校准
    pcl::SACSegmentation<pcl::PointXYZ> plane_seg;
    pcl::PointIndices::Ptr plane_inliers ( new pcl::PointIndices );
    pcl::ModelCoefficients::Ptr plane_coefficients ( new pcl::ModelCoefficients );
    plane_seg.setOptimizeCoefficients (true);
    plane_seg.setModelType ( pcl::SACMODEL_PLANE );
    plane_seg.setMethodType ( pcl::SAC_RANSAC );
    plane_seg.setDistanceThreshold ( 0.01 );
    plane_seg.setInputCloud ( map_read );
    plane_seg.segment (*plane_inliers, *plane_coefficients);//得到平面系数，进而得到平面法向量

    Eigen::Vector3f plane_normal;
    plane_normal << plane_coefficients->values[0], plane_coefficients->values[1], plane_coefficients->values[2];
    cout<<CYAN<<"plane_normal "<<endl<<plane_normal<<RESET<<endl;

    if(allow_plane_align)
    {
        //是否允许进行变换
        Eigen::Matrix4f transform_x = Eigen::Matrix4f::Identity();
        /* x转移矩阵
        |1   0   0    |
        |0  cos -sin  |
        |0  sin  cos  |
        */
        double cosx, sinx;
        double dis = sqrt(pow(plane_normal[0], 2) + pow(plane_normal[1], 2) + pow(plane_normal[2], 2));
        cosx = sqrt(pow(plane_normal[0], 2) + pow(plane_normal[2], 2))/dis;
        sinx = plane_normal[1] / dis;
        transform_x(1, 1) = cosx;
        transform_x(1, 2) = -sinx;
        transform_x(2,1) = sinx;
        transform_x(2,2) = cosx;

        Eigen::Matrix4f transform_y = Eigen::Matrix4f::Identity();
        /* y转移矩阵
        |cos  0  sin  |
        |0    1   0   |
        |-sin 0  cos  |
        */
        double cosy, siny;
        cosy = plane_normal[2]/dis;
        siny = sqrt(pow(plane_normal[0],2)+pow(plane_normal[1],2))/dis;
        // Define a rotation matrix (see https://en.wikipedia.org/wiki/Rotation_matrix)
        transform_y(0, 0) = cosy;
        transform_y(0, 2) = siny;
        transform_y(2, 0) = -siny;
        transform_y(2, 2) = cosy;

        align_rotation_inverse = transform_x*transform_y;
        pcl::transformPointCloud(*map_read, *map_read, align_rotation_inverse); //对map_read先进行变换
    }

    //确定分辨率
    int w = map_read->width ;
    int h = map_read->height;
    std::cout << "width "
              << map_read->width <<" height "<< map_read->height // 宽*高
              << std::endl;
    //划分体素格子
    pcl::PointXYZ min, max;
    
    pcl::getMinMax3D(*map_read, min, max); //查找点云的x，y，z方向的极值
    
    /// 方式2：Affine3f
	// 创建矩阵对象transform_2.matrix()，初始化为4×4单位阵
	// Eigen::Affine3f inner_transform = Eigen::Affine3f::Identity();
	// 定义平移
    // float x_width = max.x- min.x;
    // float y_width = max.y- min.y;
    // float z_width = max.z- min.z;
    cout << "original map " <<endl;
    cout << "->min_x = " << min.x << endl;
	cout << "->min_y = " << min.y << endl;
	cout << "->min_z = " << min.z << endl;
	cout << "->max_x = " << max.x << endl;
	cout << "->max_y = " << max.y << endl;
	cout << "->max_z = " << max.z << endl;

    static int times = 0;
    float square_size = 20.0f;
    float square_gap = 2.0f;
    short x_num = ceil(((max.x-min.x) - square_size)/square_gap) +1;
    short y_num = ceil(((max.y-min.y) - square_size)/square_gap) +1;
    cout<<BLUE<<"creating a submap of "<<x_num<<"*"<<y_num<<" with square_size="<<square_size <<"m and square_gap="<<square_gap<<"m"<<RESET<<endl;

    //outfile<<x_num<< " "<<y_num<<" "<<square_size<<" "<<square_gap<<" "<<endl; //保存到.txt

    float this_x_left;
    float this_x_right;
    float this_y_left;
    float this_y_right;
    pcl::PointCloud<pcl::PointXYZ>::Ptr map_filtered(new pcl::PointCloud<pcl::PointXYZ>);
    // 设置滤波器对象
    pcl::PassThrough<pcl::PointXYZ> pass;

    for(int x_idx = 0; x_idx< x_num; x_idx++ )
    {
        cout<<YELLOW<<"x is "<<x_idx<<"/"<<x_num<<RESET<<endl;
        #pragma omp parallel num_threads(8)
        #pragma omp parallel for    
        for(int y_idx = 0; y_idx< y_num; y_idx++ )
        {
            int this_idx = y_idx + x_idx*y_num; //每循环一次x，就有y_num个被数过了
            
            this_x_left = min.x + square_gap*x_idx;
            this_x_right = min.x + square_gap*x_idx + square_size;
            this_y_left = min.y + square_gap*y_idx;
            this_y_right = min.y + square_gap*y_idx + square_size;

            pass.setInputCloud (map_read);            //设置输入点云
            pass.setFilterFieldName ("x");         //设置过滤时所需要点云类型的Z字段
            pass.setFilterLimits (this_x_left, this_x_right);        //设置在过滤字段的范围
            //pass.setFilterLimitsNegative (true);   //设置保留范围内还是过滤掉范围内
            pass.filter (*map_filtered);            //执行滤波，保存过滤结果在cloud_filtered

            pass.setInputCloud (map_filtered);            //设置输入点云
            pass.setFilterFieldName ("y");         //设置过滤时所需要点云类型的Z字段
            pass.setFilterLimits (this_y_left, this_y_right);        //设置在过滤字段的范围
            //pass.setFilterLimitsNegative (true);   //设置保留范围内还是过滤掉范围内
            pass.filter (*map_filtered);            //执行滤波，保存过滤结果在cloud_filtered

            if(map_filtered->points.size() >= 20 ) //点云较大
            {
                cout<<"this_idx "<<this_idx<<endl;
                pcl::io::savePCDFileASCII(PcdSavePrefix + to_string(this_idx) + ".pcd", *map_filtered);
                // std::string tmp_str = to_string(this_idx) + ".txt";
                // std::string bin_file = BinSavePrefix+ tmp_str;
                // //std::cout << bin_file << std::endl;
                // PCLtoBin(map_filtered,bin_file);
            }
        }
    }

    // read_filelists( PcdSavePrefix, file_lists, "pcd" );
    // //sort_filelists(file_lists, "pcd" );
    // for (int i = 0; i < file_lists.size(); ++i)
    // {
    //     std::string pcd_file = PcdSavePrefix + file_lists[i];
    //     std::string tmp_str = file_lists[i].substr(0, file_lists[i].length() - 4) + ".bin";
    //     std::string bin_file = BinSavePrefix+ tmp_str;
    //     std::cout << bin_file << std::endl;
    //     convertPCDtoBin( pcd_file, bin_file);
    // }


    // 似乎没有必要转换一次
    // Eigen::Affine3f minmax_transform = Eigen::Affine3f::Identity();
	// minmax_transform.translation() << -min.x 
    //                                  , -min.y 
    //                                  , -min.z ;	// 三个数分别对应X轴、Y轴、Z轴方向上的平移
	// // 定义旋转矩阵，绕z轴
	// minmax_transform.rotate(Eigen::AngleAxisf(0, Eigen::Vector3f::UnitZ()));	//同理，UnitX(),绕X轴；UnitY(),绕Y轴.

	// pcl::transformPointCloud(*map_read, *Mmap, minmax_transform);	//再平移到全部为正

    // inner_transform =  minmax_transform * align_rotation_inverse; //这是处理阶段用的两个两个变换

    // // cout<<"1-2"<<endl;
    // cout<<"transformed map"<<endl;

    // pcl::getMinMax3D(*Mmap, min, max); //查找点云的x，y，z方向的极值

    // cout << "->min_x = " << min.x << endl;
	// cout << "->min_y = " << min.y << endl;
	// cout << "->min_z = " << min.z << endl;
	// cout << "->max_x = " << max.x << endl;
	// cout << "->max_y = " << max.y << endl;
	// cout << "->max_z = " << max.z << endl;


    // CommandLineArgs cmd_args(argc, argv);

    // // Create _outputFile folder if not exist
    // struct stat sb;
    // std::string folderPath = cmd_args._pcd_path;
    // if (! (stat(folderPath.c_str(), &sb) == 0 && S_ISDIR(sb.st_mode)) )
    // {//...It is not a directory...
    //     mkdir(folderPath.c_str(), 0755);
    // }
    // folderPath = cmd_args._bin_path;
    // if (! (stat(folderPath.c_str(), &sb) == 0 && S_ISDIR(sb.st_mode)) )
    // {//...It is not a directory...
    //     mkdir(folderPath.c_str(), 0755);
    // }

    

    // if(cmd_args._mode == "bin2pcd")
    // {
    //     read_filelists( cmd_args._bin_path, file_lists, "bin" );
    //     sort_filelists( file_lists, "bin" );

    //     #pragma omp parallel num_threads(8)
    //     #pragma omp parallel for
    //     for (int i = 0; i < file_lists.size(); ++i)
    //     {
    //         std::string bin_file = cmd_args._bin_path + file_lists[i];
    //         std::string tmp_str = file_lists[i].substr(0, file_lists[i].length() - 4) + ".pcd";
    //         std::string pcd_file = cmd_args._pcd_path + tmp_str;
    //         readKittiPclBinData( bin_file, pcd_file );
    //     }
    // } 

    

    
    outfile.close();
    return 0;
}
