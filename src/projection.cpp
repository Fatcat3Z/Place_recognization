//
// Created by FATCAT.STARK on 2021/10/4.
//
#include "projection.h"
#include <ctime>

using namespace std;
using namespace cv;
using namespace pcl;

void RangeProjection::frontproject(vector<pcl::PointCloud<pcl::PointXYZ>::Ptr>& Eucluextra) {
    clock_t startTime,endTime;
    startTime = clock();
    // 雷达参数设置,上下视角转弧度制
    double fov_up = _fov_up / 180.0 * CV_PI;
    double fov_down = _fov_down / 180.0 * CV_PI;
    double fov = abs(fov_up) + abs(fov_down);

    // 计算点云的各项参数
    Mat range_mat = Mat::zeros(_proj_H, _proj_W, CV_32FC1);
    Mat spatial_area_mat = Mat::zeros(_proj_H, _proj_W, CV_32FC1);
    Mat depth_order_mat = Mat::zeros(_proj_H, _proj_W, CV_32FC1);
    boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer(new pcl::visualization::PCLVisualizer(" cloud viewer"));
    if(_showclouds){
        viewer->setBackgroundColor(0, 0, 0);
        viewer->addCoordinateSystem(1.0);
    }
    int i = 0;
    vector<pair<pcl::PointCloud<pcl::PointXYZ>::Ptr, double>> cloudndepth;
    vector<pair<pcl::PointXYZ, double>> centroidsndepth;
    map<PointXYZ, vector<double>> pointnorder;
    vector<vector<double>> cloud_segments;
    int segments_num = Eucluextra.size();
    cout<<"segments_num"<<segments_num;

    // 这里的点云段还没有按照距离远近进行排序
    for(auto& cloud: Eucluextra){
        pcl::PointXYZ centroid = extractsegments::calculate_centroid(cloud);
        double range = pcl::euclideanDistance(PointXYZ(0,0,0), centroid);
        cloudndepth.emplace_back(make_pair(cloud, range));
        centroidsndepth.emplace_back(make_pair(centroid, range));
        for(const auto& point: *cloud){
            double depth = pcl::euclideanDistance(PointXYZ(0,0,0), point);
            cloud_segments.push_back({point.x, point.y, point.z, depth});
            pointnorder.insert(make_pair(point, 0));
        }

        string str_filename = "cloud_segmented" + to_string(i);
        pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> color(cloud, random() % 255, random() % 255, random() % 255);
        i++;
        if(_showclouds) viewer->addPointCloud(cloud, color, str_filename);
    }
    if(_showclouds) viewer->spin();
    // 对所有点云按照距离远近降序排列
    sort(cloud_segments.begin(), cloud_segments.end(), comparevecdep);
    // 将所有段的质心按照距离降序排列，计算最近邻面积
    sort(centroidsndepth.begin(), centroidsndepth.end(), comparecentroiddep);
    vector<double> spatial_areas = extractor.calculate_spatial_area(centroidsndepth, _topk);
    // 对所有的段按照质心距离远近降序排列,并赋予距离次序以及
    sort(cloudndepth.begin(), cloudndepth.end(), compareclouddep);
    double depth_order = 1.0;
    for(auto& cloud: cloudndepth){
        for(const auto& point: *(cloud.first)){
            pointnorder[point][0] = depth_order;
            pointnorder[point][1] = spatial_areas[depth_order-1];
        }
        depth_order++;
    }
    // 已经按照距离由远到近进行段排序，使距离近的点云能够在投影时覆盖距离元的点云
    cout<<"project points:"<<cloud_segments.size()<<endl;
    for(const auto& point: cloud_segments){
        pcl::PointXYZ targetpoint(point[0], point[1], point[2]);
        double depth = point[3];
        cout<<"depth:"<<depth<<endl;
        if(depth > _min_range && depth < _max_range){
            // 计算偏航角和俯仰角
            double yaw = - atan2(point[1], point[0]);
            double pitch = asin(point[2] / depth);
            // cout<<"depth:"<<(depth / _max_range) * 255<<endl;
            double proj_x = 0.5 * (yaw / CV_PI + 1.0);
            double proj_y = 1.0 - (pitch + abs(fov_down)) / fov;
            proj_x *= _proj_W;
            proj_y *= _proj_H;
            // 确保投影的上下界，防止溢出
            proj_x = floor(proj_x);
            proj_y = floor(proj_y);
            proj_x = proj_x > _proj_W - 1 ? _proj_W - 1 : proj_x;
            proj_x = proj_x > 0 ? proj_x : 0;
            proj_y = proj_y > _proj_H - 1 ? _proj_H - 1 : proj_y;
            proj_y = proj_y > 0 ? proj_y : 0;

            range_mat.at<float>(int(proj_y), int(proj_x)) = _min_range / depth;
            depth_order_mat.at<float>((int)proj_y, (int)proj_x) = pointnorder[targetpoint][0] / segments_num;
            spatial_area_mat.at<float>((int)proj_y, (int)proj_x) = pointnorder[targetpoint][1];
        }
    }
    endTime = clock();
    cout << "The run time is: " <<(double)(endTime - startTime) / CLOCKS_PER_SEC << "s" << endl;
    // cout<<depth_order_mat<<endl;
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

bool RangeProjection::comparevecdep(const std::vector<double>& cloudA,
                                    const std::vector<double>& cloudB) {
    return cloudA[3] > cloudB[3];
}

bool RangeProjection::compareclouddep(const pair<pcl::PointCloud<pcl::PointXYZ>::Ptr, double> &cloudA,
                                      const pair<pcl::PointCloud<pcl::PointXYZ>::Ptr, double> &cloudB) {
    return cloudA.second > cloudB.second;
}

bool
RangeProjection::comparecentroiddep(const pair<pcl::PointXYZ, double> &cloudA,
                                    const pair<pcl::PointXYZ, double> &cloudB) {
    return cloudA.second > cloudB.second;
}




