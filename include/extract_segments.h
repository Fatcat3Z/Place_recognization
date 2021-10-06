//
// Created by FATCAT.STARK on 2021/10/5.
//

#ifndef PLACE_RECOGNIZATION_EXTRACT_SEGMENTS_H
#define PLACE_RECOGNIZATION_EXTRACT_SEGMENTS_H
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/filter_indices.h>
#include <pcl/features/normal_3d.h>
#include <pcl/kdtree/kdtree.h>
#include <pcl/kdtree/impl/kdtree_flann.hpp>
#include <pcl/search/impl/kdtree.hpp>
#include <pcl/ModelCoefficients.h>
#include <pcl/visualization/cloud_viewer.h>
#include <pcl/visualization/pcl_visualizer.h>
//#include <opencv2/opencv.hpp>

class extractsegments{
public:
    explicit extractsegments(float sensor_height=1.5):
    _clustertolerance(0.2),
    _minclustersize(100),
    _maxclustersize(15000),
    _fliter_max_iteration(100),
    _distance_threshold(0.2),
    _sensor_height(sensor_height){};
    std::vector<pcl::PointIndices> extract_cluster_indices(const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud_filtered);
    void filtercloud(pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud, pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud_filtered);
private:
    float _clustertolerance;      // 最近邻搜索范围
    int _minclustersize;        // 每个聚类的最小点云数
    int _maxclustersize;        // 每个聚类的最大点云数
    int _fliter_max_iteration;  // 表示点到估计模型的距离最大值
    float _distance_threshold;  // 滤除地面时，设定的距离阀值，距离阀值决定了点被认为是局内点是必须满足的条件
    float _sensor_height;
};

#endif //PLACE_RECOGNIZATION_EXTRACT_SEGMENTS_H
