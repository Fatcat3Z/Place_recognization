//
// Created by FATCAT.STARK on 2021/10/5.
//
#ifndef PLACE_RECOGNIZATION_EXTRACT_SEGMENTS_H
#define PLACE_RECOGNIZATION_EXTRACT_SEGMENTS_H
#include <opencv2/opencv.hpp>
#include "pcl_utils.h"


class extractsegments{
public:
    explicit extractsegments(bool show = false, float sensor_height=1.8):
    _show(show),
    _clustertolerance(0.2),
    _minclustersize(100),
    _maxclustersize(15000),
    _fliter_max_iteration(100),
    _distance_threshold(0.2),
    _normal_dist_weight(0.1),
    _max_spatial_distance(5),
    _filter_flat_seg(false),
    _horizontal_ratio(15),
    _sensor_height(sensor_height){};
    std::vector<pcl::PointIndices> extract_cluster_indices(const pcl::PointCloud<PointType>::Ptr& cloud_filtered);
    void filtercloud(pcl::PointCloud<PointType>::Ptr &cloud, pcl::PointCloud<PointType>::Ptr &cloud_filtered, bool isvoxeled);
    static PointType calculate_centroid(pcl::PointCloud<PointType>::Ptr& cloud);
    static double calculate_area_of_triangle(const PointType& pointa, const PointType& pointb, const PointType& pointc);
    std::vector<double> calculate_spatial_area(const std::vector<std::pair<PointType, double>>& centroids, int topk);
    static bool comparedepth(const std::pair<PointType, double>& pointdepthA, const std::pair<PointType, double>& pointdepthB);
    void show_senmantic_points(const pcl::PointCloud<PointType>::Ptr& cloud2show, const std::vector<int>& labels);
    std::vector<pcl::PointCloud<PointType>::Ptr> extract_segments(const pcl::PointCloud<PointType>::Ptr &cloud_filtered);

private:
    float _clustertolerance;      // 最近邻搜索范围
    int _minclustersize;          // 每个聚类的最小点云数
    int _maxclustersize;          // 每个聚类的最大点云数
    int _fliter_max_iteration;    // 表示点到估计模型的距离最大值
    float _distance_threshold;    // 滤除地面时，设定的距离阀值，距离阀值决定了点被认为是局内点是必须满足的条件
    float _normal_dist_weight;    // 地平面提取的法向距离权重
    float _sensor_height;
    float _max_spatial_distance;  // 最远空间特征距离（组成三角形）
    bool _show;
    bool _filter_flat_seg;
    double _horizontal_ratio;
};

#endif //PLACE_RECOGNIZATION_EXTRACT_SEGMENTS_H
