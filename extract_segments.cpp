//
// Created by FATCAT.STARK on 2021/10/5.
//
#include "extract_segments.h"

using namespace std;

std::vector<pcl::PointIndices> extractsegments::extract_cluster_indices(const pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud_filtered) {
    vector<pcl::PointIndices> cluster_indices;
    pcl::search::KdTree<pcl::PointXYZ>::Ptr kdtree(new pcl::search::KdTree<pcl::PointXYZ>);
    kdtree->setInputCloud(cloud_filtered);

    pcl::EuclideanClusterExtraction<pcl::PointXYZ> ec;   // 创建欧式聚类分割对象
    ec.setClusterTolerance(_clustertolerance);           // 设置近邻搜索的搜索半径
    ec.setMinClusterSize(_minclustersize);               // 设置最小聚类尺寸
    ec.setMaxClusterSize(_maxclustersize);
    ec.setSearchMethod(kdtree);
    ec.setInputCloud(cloud_filtered);
    ec.extract(cluster_indices);

    vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> Eucluextra; // 用于储存欧式分割后的点云
    for(auto iter = cluster_indices.begin(); iter != cluster_indices.end(); iter++){
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_cluster(new pcl::PointCloud<pcl::PointXYZ>);
        for(auto pit = iter->indices.begin(); pit != iter->indices.end(); pit++)
            cloud_cluster->points.push_back(cloud_filtered->points[*pit]);
        cloud_cluster->width = cloud_cluster->points.size();
        cloud_cluster->height = 1;
        cloud_cluster->is_dense = true;
        Eucluextra.push_back(cloud_cluster);

    }
    return cluster_indices;
}

void extractsegments::filtercloud(pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud, pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud_filtered) {
    if (!cloud->empty()){
        // 先体素化再作平面分割
        pcl::VoxelGrid<pcl::PointXYZ> vg; //体素栅格下采样对象
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_vg (new pcl::PointCloud<pcl::PointXYZ>);
        vg.setInputCloud (cloud);
        vg.setLeafSize (0.01f, 0.01f, 0.01f); //设置采样的体素大小
        vg.filter (*cloud_vg);  //执行采样保存数据

        //创建分割时所需要的模型系数对象，coefficients及存储内点的点索引集合对象inliers
        pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients);
            pcl::PointIndices::Ptr inliers(new pcl::PointIndices);
        // 创建分割对象
        pcl::SACSegmentation<pcl::PointXYZ> seg;
        // 可选择配置，设置模型系数需要优化
        seg.setOptimizeCoefficients(true);
        // 必要的配置，设置分割的模型类型，所用的随机参数估计方法，距离阀值，输入点云
        seg.setModelType(pcl::SACMODEL_PLANE);      // 设置模型类型
        seg.setMethodType(pcl::SAC_RANSAC);                // 设置随机采样一致性方法类型
        seg.setMaxIterations(_fliter_max_iteration);
        seg.setDistanceThreshold(_distance_threshold);
        seg.setInputCloud(cloud_vg);
        //引发分割实现，存储分割结果到点几何inliers及存储平面模型的系数coefficients
        seg.segment(*inliers, *coefficients);
        if (inliers->indices.empty())
        {
            cout<<"error! Could not found any inliers!"<<endl;
        }
        // extract ground
        // 从点云中抽取分割的处在平面上的点集
        pcl::ExtractIndices<pcl::PointXYZ> extractor;//点提取对象
        extractor.setInputCloud(cloud);
        extractor.setIndices(inliers);
        extractor.setNegative(true);        // true 表示滤除地面 false表示提取地面
        extractor.filter(*cloud_filtered);
        // vise-versa, remove the ground not just extract the ground
        // just setNegative to be true
        cout << "filter done."<<endl;
        pcl::visualization::CloudViewer viewer("Cloud viewer");
        viewer.showCloud(cloud_filtered);

    }else{
        cout<<"no raw PointCloud data!"<<endl;
    }
}
