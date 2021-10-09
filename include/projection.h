//
// Created by FATCAT.STARK on 2021/10/4.
//

#ifndef PLACE_RECOGNIZATION_PROJECTION_H
#define PLACE_RECOGNIZATION_PROJECTION_H
#include <iostream>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <opencv2/opencv.hpp>
#include <vector>
#include <cmath>
#include <utility>
#include <map>
#include "extract_segments.h"

struct map_compare{
    bool operator ()(const pcl::PointXYZ &pointA, const pcl::PointXYZ &pointB) const{
        return pointA.x > pointB.x;
    };

};
class RangeProjection{

public:
    RangeProjection(bool showclouds = false, bool showprojections = true, double fov_up=3.0, double fov_down=-25.0, int proj_H=64, int proj_W=900, float max_range=50.0)
    :_showclouds(showclouds),
    _showprojectios(showprojections),
    _fov_up(fov_up),
    _fov_down(fov_down),
    _proj_H(proj_H),
    _proj_W(proj_W),
    _max_range(max_range),
    _min_range(3.0),
    _topk(5),
    _scan_sector(900),
    _scan_ring(64),
    _scan_top(4),
    _sensor_height(1.8){};

    void getprojection(std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr>& Eucluextra);
    extractsegments extractor;
    cv::Mat frontprojection(const std::vector<std::vector<double>>& cloud_segments, std::map<pcl::PointXYZ, std::vector<double>, map_compare> pointnorder, int state);
    static bool compareclouddep(const std::pair<pcl::PointCloud<pcl::PointXYZ>::Ptr, double>& cloudA,
                                const std::pair<pcl::PointCloud<pcl::PointXYZ>::Ptr, double>& cloudB);
    static bool comparecentroiddep(const std::pair<pcl::PointXYZ, double>& cloudA,
                                   const std::pair<pcl::PointXYZ, double>& cloudB);
    static bool comparevecdep(const std::vector<double>& cloudA,
                              const std::vector<double>& cloudB);
    void set_segment_num(int segments){ _segments = segments;}
    cv::Mat scancontext(const std::vector<std::vector<double>>& cloud_segments);

private:
    int _segments;
    double _fov_up;        // 正向角度视野   C32为15
    double _fov_down;      // 负向角度视野   C32为-16
    int _proj_H;        // 投影伪图像的行数，对应于垂直分辨率
    int _proj_W;        // 投影伪图像的列数，对应于水平分辨率
    float _max_range;     // 深度的最大有效距离
    float _min_range;     // 深度的最近有效距离
    int _topk;
    bool _showclouds;
    bool _showprojectios;
    int _scan_sector;
    int _scan_ring;
    int _scan_top;
    float _sensor_height;
};
#endif //PLACE_RECOGNIZATION_PROJECTION_H
