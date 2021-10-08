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
#include "extract_segments.h"

class RangeProjection{

public:
    RangeProjection(bool showclouds = false, bool showprojections = true, int fov_up=3, int fov_down=-25, int proj_H=64, int proj_W=900, int max_range=30)
    :_showclouds(showclouds),
    _showprojectios(showprojections),
    _fov_up(fov_up),
    _fov_down(fov_down),
    _proj_H(proj_H),
    _proj_W(proj_W),
    _max_range(max_range),
    _topk(5){};

    void frontproject(std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr>& Eucluextra);
    extractsegments extractor;
    static bool comparedepth(const std::pair<pcl::PointCloud<pcl::PointXYZ>::Ptr, double>& cloudA, const std::pair<pcl::PointCloud<pcl::PointXYZ>::Ptr, double>& cloudB);
    static bool comparedep(const std::pair<pcl::PointXYZ, double>& centroidA, const std::pair<pcl::PointXYZ, double>& centroidB);

private:
    int _segments_num;  // 输入的段落数量
    int _fov_up;        // 正向角度视野   C32为15
    int _fov_down;      // 负向角度视野   C32为-16
    int _proj_H;        // 投影伪图像的行数，对应于垂直分辨率
    int _proj_W;        // 投影伪图像的列数，对应于水平分辨率
    int _max_range;     // 深度的最大有效距离
    int _topk;
    bool _showclouds;
    bool _showprojectios;
};
#endif //PLACE_RECOGNIZATION_PROJECTION_H
