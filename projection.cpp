//
// Created by FATCAT.STARK on 2021/10/4.
//
#include "projection.h"

using namespace std;
using namespace cv;
using namespace pcl;

void RangeProjection::sortdepth(std::vector<cv::Point3d> &points) {
    // 雷达参数设置,上下视角转弧度制
    double fov_up = _fov_up / 180.0 * CV_PI;
    double fov_down = _fov_down / 180.0 * CV_PI;
    double fov = abs(fov_up) + abs(fov_down);

    // 计算点云的各项参数
    vector<pair<Point3d, double>> pointsndeph;
    vector<vector<double>> pointparamters;
    for(const auto& point: points){
        double depth = sqrt(point.x * point.x + point.y * point.y + point.z * point.z);
        // 计算偏航角和俯仰角
        double yaw = - atan2(point.y, point.x);
        double pitch = asin(point.z / depth);

        double proj_x = 0.5 * (yaw / CV_PI + 1.0);
        double proj_y = 1.0 - (pitch + abs(_fov_down)) / fov;
        proj_x *= _proj_W;
        proj_y *= _proj_H;
        // 确保投影的上下界，防止溢出
        proj_x = proj_x > _proj_W - 1 ? _proj_W - 1 : proj_x;
        proj_x = proj_x > 0 ? proj_x : 0;
        proj_y = proj_y > _proj_H - 1 ? _proj_H - 1 : proj_y;
        proj_y = proj_y > 0 ? proj_y : 0;

        pair<Point3d, double> pointndepth = make_pair(point, depth);
        pointsndeph.push_back(pointndepth);
        pointparamters.push_back({depth, proj_x, proj_y});
    }
    sort(pointsndeph.begin(), pointsndeph.end(), comparedepth);


}

bool RangeProjection::comparedepth(const pair<cv::Point3d, double> &pointndepthA,
                                   const pair<cv::Point3d, double> &pointndepthB) {
    return pointndepthA.second > pointndepthB.second;
}


