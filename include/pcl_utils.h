//
// Created by FATCAT.STARK on 2022/3/18.
//

#ifndef PLACE_RECOGNIZATION_PCL_UTILS_H
#define PLACE_RECOGNIZATION_PCL_UTILS_H
#include <Eigen/Core>
#include <pcl/common/common.h>
#include <pcl/common/transforms.h>
#include <pcl/common/distances.h>
#include <pcl/pcl_macros.h>
#include <string>
#include <iostream>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/kdtree/kdtree.h>
#include <pcl/kdtree/impl/kdtree_flann.hpp>
#include <pcl/search/kdtree.h>
#include <pcl/features/moment_of_inertia_estimation.h>
#include <pcl/features/normal_3d.h>
#include <pcl/filters/passthrough.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/filters/radius_outlier_removal.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/filters/filter_indices.h>
#include <pcl/io/pcd_io.h>
#include <pcl/io/ply_io.h>
#include <pcl/ModelCoefficients.h>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/visualization/cloud_viewer.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/segmentation/impl/sac_segmentation.hpp>
#include <pcl/search/impl/kdtree.hpp>
#include <string>
#include <queue>


struct SemanticPoint
{
    PCL_ADD_POINT4D;
    PCL_ADD_INTENSITY;
    int senmantic;
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
} EIGEN_ALIGN16;

POINT_CLOUD_REGISTER_POINT_STRUCT (SemanticPoint,
                                   (float, x, x)
                                           (float, y, y)
                                           (float, z, z)
                                           (float, intensity, intensity)
                                           (int, senmantic, senmantic)
)

typedef SemanticPoint PointType;

std::vector<std::string> getFiles(const std::string& path){
    std::vector<std::string> filenames;
    DIR *pDir;
    struct dirent* ptr;
    if(!(pDir = opendir(path.c_str()))){
        std::cout<<"Folder "<< path.c_str() <<" doesn't Exist!"<<std::endl;
        return filenames;
    }
    while((ptr = readdir(pDir))!=0) {
        if (strcmp(ptr->d_name, ".") != 0 && strcmp(ptr->d_name, "..") != 0){
            filenames.push_back(path + "/" + ptr->d_name);
        }
    }
    closedir(pDir);
    sort(filenames.begin(), filenames.end());
    return filenames;
}

void pcl_loader(const std::string& pcd_file, pcl::PointCloud<PointType>::Ptr &outCloud) {
    // loader
    pcl::io::loadPCDFile(pcd_file, *outCloud);
    pcl::PassThrough<PointType> pass;
    pass.setInputCloud(outCloud);
    pass.setFilterFieldName("intensity");
    pass.setFilterLimits(1, 255);
    pass.setFilterLimitsNegative(false);
    pass.filter(*outCloud);
}

void pcl_load_semantic(const std::string& bin_file, pcl::PointCloud<SemanticPoint>::Ptr &outCloud){
    std::ifstream inFile_3d;
    inFile_3d.open(bin_file);
    if (!inFile_3d.is_open()) {
        std::cout << "Can not open the real 3d Points file " << bin_file << std::endl;
        exit(1);
    }
    std::string linestr;
    while (getline(inFile_3d, linestr)) {
        std::vector<std::string> strvec;
        std::string s;
        std::stringstream ss(linestr);
        while (getline(ss, s, ','))
            strvec.push_back(s);
        SemanticPoint this_point;
        this_point.x = atof(strvec[0].c_str());
        this_point.y = atof(strvec[1].c_str());
        this_point.z = atof(strvec[2].c_str());
        this_point.intensity = atof(strvec[3].c_str());
        this_point.senmantic = atoi(strvec[4].c_str());
        outCloud->push_back(this_point);
    }
}
void pcl_loader(const std::string& pcd_file, pcl::PointCloud<pcl::PointXYZ>::Ptr &outCloud) {
    // loader
    pcl::io::loadPCDFile(pcd_file, *outCloud);
}


void load_pcd(const std::string& pcd_path,pcl::PointCloud<PointType>::Ptr& input_cloud){
    if(pcd_path.substr(pcd_path.size()-4,4)!=".pcd"){
        std::vector<std::string> cur_pcd_list = getFiles(pcd_path);
        for(int j=0;j<cur_pcd_list.size();j++){
            pcl::PointCloud<PointType>::Ptr cur_input_cloud(new pcl::PointCloud<PointType>);
            pcl_loader(cur_pcd_list[j], cur_input_cloud);

            *input_cloud=*input_cloud + *cur_input_cloud;
        }
    }else{
        pcl_loader(pcd_path, input_cloud);
    }
}

void load_pcd(const std::string& pcd_path,pcl::PointCloud<pcl::PointXYZ>::Ptr& input_cloud){
    if(pcd_path.substr(pcd_path.size()-4,4)!=".pcd"){
        std::vector<std::string> cur_pcd_list = getFiles(pcd_path);
        for(int j=0;j<cur_pcd_list.size();j++){
            pcl::PointCloud<pcl::PointXYZ>::Ptr cur_input_cloud(new pcl::PointCloud<pcl::PointXYZ>);
            pcl_loader(cur_pcd_list[j], cur_input_cloud);

            *input_cloud=*input_cloud + *cur_input_cloud;
        }
    }else{
        pcl_loader(pcd_path, input_cloud);
    }
}

void lidar_fg_search(const pcl::PointCloud<PointType>::Ptr &background,
                     const pcl::PointCloud<PointType>::Ptr &inputground,
                     pcl::PointCloud<PointType>::Ptr &fgcloud,
                     pcl::PointCloud<PointType>::Ptr &bkcloud){
    const float search_radius = 0.2;
    const int fg_thresh = 0.5;

    pcl::KdTreeFLANN<PointType> kdtree;
    kdtree.setInputCloud(background);
    std::vector<int> radiusSearch_idx;
    std::vector<float> radiusSearch_distance;
    fgcloud->clear();
    bkcloud->clear();

    std::vector<bool> fg_flag(inputground->size());

    for (int i=0;i<inputground->size();++i) {
        const PointType& thisPt = inputground->points[i];
        if (kdtree.radiusSearch(thisPt, search_radius, radiusSearch_idx, radiusSearch_distance) > 0) {
            if (radiusSearch_idx.size() < fg_thresh) {
                fg_flag[i] = true;
            }
            else {
                fg_flag[i] = false;
            }
        }
        else {
            fg_flag[i] = true;
        }
    }
    for (int i=0;i<fg_flag.size();++i) {
        if (fg_flag[i])
            fgcloud->points.push_back(inputground->points[i]);
        else
            bkcloud->points.push_back(inputground->points[i]);
    }
}

void pcl_euclidean_cluster(const pcl::PointCloud<PointType>::Ptr &cloud,
                           std::vector<pcl::PointIndices> &cluster_indices) {
    float search_radius = 0.1;
    int min_cluster_size = 100;
    int max_cluster_size = 15000;

    pcl::search::KdTree<PointType>::Ptr kdtree(new pcl::search::KdTree<PointType>);
    kdtree->setInputCloud(cloud);
    pcl::EuclideanClusterExtraction<PointType> cluster;
    cluster.setClusterTolerance(search_radius);
    cluster.setMinClusterSize(min_cluster_size);
    cluster.setMaxClusterSize(max_cluster_size);
    cluster.setSearchMethod(kdtree);
    cluster.setInputCloud(cloud);
    cluster.extract(cluster_indices);
}


void lidar_invasion_fg_cluster(const pcl::PointCloud<PointType>::Ptr &fgCloud,
                               std::vector<pcl::PointCloud<PointType>::Ptr> &fgCloud_list) {

    std::vector<pcl::PointIndices> cluster_indices;
    pcl_euclidean_cluster(fgCloud, cluster_indices);
    for (auto &this_indice: cluster_indices) {
        pcl::PointCloud<PointType>::Ptr this_cluster(new pcl::PointCloud<PointType>);
        for (int &thsIdx : this_indice.indices) {
            this_cluster->push_back(fgCloud->points[thsIdx]);
        }
        fgCloud_list.push_back(this_cluster);
    }
}



#endif //PLACE_RECOGNIZATION_PCL_UTILS_H
