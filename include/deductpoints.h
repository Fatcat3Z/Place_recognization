//
// Created by wy-lab on 2021/10/18.
//

#ifndef PLACE_RECOGNIZATION_DEDUCTPOINTS_H
#define PLACE_RECOGNIZATION_DEDUCTPOINTS_H
#include "extract_segments.h"

class deductpts{
public:
    std::vector<std::vector<float>> loadboxes(const std::string& boxespath);
    bool isinbox(const std::vector<float> &boxparamters, const pcl::PointXYZ& checkpoint);
    int findnearestbox(const std::vector<std::vector<float>> &boxparamters, const pcl::PointXYZ& checkpoint);
    Eigen::Quaternionf getquaterion(const float& angle);
    pcl::PointXYZ rotatept(const pcl::PointXYZ& targetpt, const float& angle);
    pcl::PointCloud<pcl::PointXYZ>::Ptr removeobjs(pcl::PointCloud<pcl::PointXYZ>::Ptr & cloud, const std::vector<std::vector<float>>& boxes);
};
#endif //PLACE_RECOGNIZATION_DEDUCTPOINTS_H
