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
    vector<pcl::PointXYZ> centroids(Eucluextra.size());
    if(_showclouds){
        viewer->setBackgroundColor(0, 0, 0);
        viewer->addCoordinateSystem(1.0);
    }
    int i = 0;
    vector<pair<pcl::PointCloud<pcl::PointXYZ>::Ptr, double>> cloudndepth;
    for(auto& cloud: Eucluextra){
        pcl::PointXYZ centroid = extractsegments::calculate_centroid(cloud);
        double range = pcl::euclideanDistance(PointXYZ(0,0,0), centroid);
        cloudndepth.emplace_back(make_pair(cloud, range));
        centroids.push_back(centroid);
        string str_filename = "cloud_segmented" + to_string(i);
        pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> color(cloud, 255 * (1 - i)*(2 - i) / 2, 255 * i*(2 - i), 255 * i*(i - 1) / 2);
        i++;
        if(_showclouds) viewer->addPointCloud(cloud, color, str_filename);
    }
    if(_showclouds) viewer->spin();
    sort(cloudndepth.begin(), cloudndepth.end(), comparedepth);
    vector<double> spatial_areas = extractor.calculate_spatial_area(centroids, _topk);
    // for(const double& area : spatial_areas) cout<<area<<endl;
    for(auto& cloud: cloudndepth){
        for(const auto& point: *(cloud.first)){
            double depth = pcl::euclideanDistance(PointXYZ(0,0,0), point);
            // 计算偏航角和俯仰角
            double yaw = - atan2(point.y, point.x);
            double pitch = asin(point.z / depth);
            cout<<"depth:"<<(depth / _max_range) * 255<<endl;
            double proj_x = 0.5 * (yaw / M_PI + 1.0);
            double proj_y = 1.0 - (pitch + abs(fov_down)) / fov;
            proj_x *= _proj_W;
            proj_y *= _proj_H;
            // 确保投影的上下界，防止溢出
            proj_x = proj_x > _proj_W - 1 ? _proj_W - 1 : proj_x;
            proj_x = proj_x > 0 ? proj_x : 0;
            proj_y = proj_y > _proj_H - 1 ? _proj_H - 1 : proj_y;
            proj_y = proj_y > 0 ? proj_y : 0;

            range_mat.at<int>((int)proj_y, (int)proj_x) = static_cast<int>((depth / _max_range) * 255);
            cout<<"(" <<proj_x<<","<< proj_y<<"):"<<range_mat.at<int>((int)proj_y, (int)proj_x)<<endl;
        }
    }
    if(_showprojectios){
         imshow("Range Projection", range_mat);
         waitKey(0);
    }

}

bool RangeProjection::comparedepth(const pair<pcl::PointCloud<pcl::PointXYZ>::Ptr, double> &cloudA,
                                   const pair<pcl::PointCloud<pcl::PointXYZ>::Ptr, double> &cloudB) {
    return cloudA.second > cloudB.second;
}




