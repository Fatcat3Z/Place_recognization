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
    RangeProjection(bool showclouds = false, bool showprojections = true, float fov_up=3.0, float fov_down=-25.0, int proj_H=64, int proj_W=900, float max_range=50.0)
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
    std::vector<cv::Mat> getprojection(std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr>& Eucluextra, bool isspatialed);
    std::vector<cv::Mat> getprojection(pcl::PointCloud<pcl::PointXYZ>::Ptr objremoved);
    extractsegments extractor;
    std::vector<cv::Mat> frontprojection(const std::vector<std::vector<float>>& cloud_segments, std::map<pcl::PointXYZ, std::vector<float>, map_compare> pointnorder, int state);
    cv::Mat getnormalmap(const cv::Mat& pointindicesmap);
    static bool compareclouddep(const std::pair<pcl::PointCloud<pcl::PointXYZ>::Ptr, float>& cloudA,
                                const std::pair<pcl::PointCloud<pcl::PointXYZ>::Ptr, float>& cloudB);
    static bool comparecentroiddep(const std::pair<pcl::PointXYZ, float>& cloudA,
                                   const std::pair<pcl::PointXYZ, float>& cloudB);
    static bool comparevecdep(const std::vector<float>& cloudA,
                              const std::vector<float>& cloudB);
    void set_segment_num(int segments){ _segments = segments;}
    cv::Mat scancontext(const std::vector<std::vector<float>>& cloud_segments);
    std::vector<cv::Mat> scancontextwithspatial(const std::vector<std::vector<float>>& cloud_segments, std::map<pcl::PointXYZ, std::vector<float>, map_compare> pointnorder, int state);
    void projectall(const std::string& rootpath, const std::vector<std::string>& filenames, const std::string& saverootpath);
    void projectsegments(const std::string& rootpath, const std::vector<std::string>& filenames, const std::string& saverootpath);
    void projectscene(const std::string& rootpath,
                      const std::vector<std::string> &filenames,
                      const std::vector<std::string>& boxfilepath,
                      const std::string &saverootpath,
                      bool segments);

private:
    int _segments;
    float _fov_up;        // ??????????????????   C32???15
    float _fov_down;      // ??????????????????   C32???-16
    int _proj_H;        // ???????????????????????????????????????????????????
    int _proj_W;        // ???????????????????????????????????????????????????
    float _max_range;     // ???????????????????????????
    float _min_range;     // ???????????????????????????
    int _topk;
    bool _showclouds;
    bool _showprojectios;
    int _scan_sector;
    int _scan_ring;
    int _scan_top;
    float _sensor_height;
};
#endif //PLACE_RECOGNIZATION_PROJECTION_H
