//
// Created by FATCAT.STARK on 2021/10/4.
//
#include "projection.h"

using namespace std;
using namespace cv;
using namespace pcl;

void RangeProjection::frontproject(vector<pcl::PointCloud<pcl::PointXYZ>::Ptr>& Eucluextra) {
    // 雷达参数设置,上下视角转弧度制
    double fov_up = _fov_up / 180.0 * M_PI;
    double fov_down = _fov_down / 180.0 * M_PI;
    double fov = abs(fov_up) + abs(fov_down);

    // 计算点云的各项参数
    Mat range_mat = Mat::zeros(_proj_H, _proj_W, CV_8UC1);
    Mat spatial_area_mat = Mat::zeros(_proj_H, _proj_W, CV_8UC1);
    Mat depth_order_mat = Mat::zeros(_proj_H, _proj_W, CV_8UC1);
    boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer(new pcl::visualization::PCLVisualizer(" cloud viewer"));
    if(_showclouds){
        viewer->setBackgroundColor(0, 0, 0);
        viewer->addCoordinateSystem(1.0);
    }
    int i = 0;
    vector<pair<pcl::PointCloud<pcl::PointXYZ>::Ptr, double>> cloudndepth;
    vector<pair<pcl::PointXYZ, double>> centroidndepth;
    int segments_num = Eucluextra.size();
    for(auto& cloud: Eucluextra){
        pcl::PointXYZ centroid = extractsegments::calculate_centroid(cloud);
        double range = pcl::euclideanDistance(PointXYZ(0,0,0), centroid);
        cloudndepth.emplace_back(make_pair(cloud, range));
        centroidndepth.emplace_back(make_pair(centroid, range));
        string str_filename = "cloud_segmented" + to_string(i);
        pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> color(cloud, 255 * (1 - i)*(2 - i) / 2, 255 * i*(2 - i), 255 * i*(i - 1) / 2);
        i++;
        if(_showclouds) viewer->addPointCloud(cloud, color, str_filename);
    }
    if(_showclouds) viewer->spin();
    sort(cloudndepth.begin(), cloudndepth.end(), comparedepth);
    vector<double> spatial_areas = extractor.calculate_spatial_area(centroidndepth, _topk);
    sort(centroidndepth.begin(), centroidndepth.end(), comparedep);
    // for(const double& area : spatial_areas) cout<<"spatial_area"<<area<<endl;
    int depth_order = 0;
    // 已经按照距离由远到近进行段排序，使距离近的点云能够在投影时覆盖距离元的点云
    for(auto& cloud: cloudndepth){
        for(const auto& point: *(cloud.first)){
            double depth = pcl::euclideanDistance(PointXYZ(0,0,0), point);
            // 计算偏航角和俯仰角
            double yaw = - atan2(point.y, point.x);
            double pitch = asin(point.z / depth);
            // cout<<"depth:"<<(depth / _max_range) * 255<<endl;
            double proj_x = 0.5 * (yaw / M_PI + 1.0);
            double proj_y = 1.0 - (pitch + abs(fov_down)) / fov;
            proj_x *= _proj_W;
            proj_y *= _proj_H;
            // 确保投影的上下界，防止溢出
            proj_x = proj_x > _proj_W - 1 ? _proj_W - 1 : proj_x;
            proj_x = proj_x > 0 ? proj_x : 0;
            proj_y = proj_y > _proj_H - 1 ? _proj_H - 1 : proj_y;
            proj_y = proj_y > 0 ? proj_y : 0;
            int pixel = static_cast<int>((depth / _max_range) * 255);
            range_mat.at<int>((int)proj_y, (int)proj_x) = pixel;
            depth_order_mat.at<int>((int)proj_y, (int)proj_x) = depth_order / segments_num * 255;
            spatial_area_mat.at<int>((int)proj_y, (int)proj_x) = spatial_areas[depth_order];
            // cout<<"(" <<proj_x<<","<< proj_y<<"):"<<range_mat.at<int>((int)proj_y, (int)proj_x)<<endl;
        }
        depth_order++;
    }
    if(_showprojectios){
        Mat spatial_projection;
        vector<Mat> final(3);
        final[0] = range_mat;
        final[1] = depth_order_mat;
        final[2] = spatial_area_mat;
        merge(final, spatial_projection);
        imshow("spatial_projection", spatial_projection);
        imshow("range projection", range_mat);
        imshow("depth order projection", depth_order_mat);
        imshow("spatial_area projection", spatial_area_mat);
        waitKey(0);
    }

}

bool RangeProjection::comparedepth(const pair<pcl::PointCloud<pcl::PointXYZ>::Ptr, double> &cloudA,
                                   const pair<pcl::PointCloud<pcl::PointXYZ>::Ptr, double> &cloudB) {
    return cloudA.second > cloudB.second;
}

bool RangeProjection::comparedep(const pair<pcl::PointXYZ, double>& centroidA,
                                   const pair<pcl::PointXYZ, double>& centroidB) {
    return centroidA.second > centroidB.second;
}




