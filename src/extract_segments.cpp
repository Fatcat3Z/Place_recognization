//
// Created by FATCAT.STARK on 2021/10/5.
//
#include "extract_segments.h"

using namespace std;

std::vector<pcl::PointIndices> extractsegments::extract_cluster_indices(const pcl::PointCloud<PointType>::Ptr &cloud_filtered) {
    vector<pcl::PointIndices> cluster_indices;
    pcl::search::KdTree<PointType>::Ptr kdtree(new pcl::search::KdTree<PointType>);
    kdtree->setInputCloud(cloud_filtered);

    pcl::EuclideanClusterExtraction<PointType> ec;   // 创建欧式聚类分割对象
    ec.setClusterTolerance(_clustertolerance);           // 设置近邻搜索的搜索半径
    ec.setMinClusterSize(_minclustersize);               // 设置最小聚类尺寸
    ec.setMaxClusterSize(_maxclustersize);
    ec.setSearchMethod(kdtree);
    ec.setInputCloud(cloud_filtered);
    ec.extract(cluster_indices);

    return cluster_indices;
}

void extractsegments::filtercloud(pcl::PointCloud<PointType>::Ptr &cloud, pcl::PointCloud<PointType>::Ptr &cloud_filtered, bool isvoxeled) {
    if (!cloud->empty()){
        pcl::search::KdTree<PointType>::Ptr tree(new pcl::search::KdTree<PointType>());

        //创建分割时所需要的模型系数对象，coefficients及存储内点的点索引集合对象inliers
        pcl::PointCloud<PointType>::Ptr cloud_remove(new pcl::PointCloud<PointType>);
        pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients);
        pcl::PointIndices::Ptr inliers(new pcl::PointIndices);
        // 创建分割对象
        pcl::NormalEstimation<PointType, pcl::Normal> ne;  //法线估计对象
        pcl::SACSegmentationFromNormals<PointType, pcl::Normal> seg;
        pcl::PointCloud<pcl::Normal>::Ptr cloud_normals(new pcl::PointCloud<pcl::Normal>);

        // 过滤后的点云进行法线估计，为后续进行基于法线的分割准备数据
        ne.setSearchMethod(tree);
        ne.setInputCloud(cloud);
        ne.setKSearch(50);
        ne.compute(*cloud_normals);

        // 可选择配置，设置模型系数需要优化
        seg.setOptimizeCoefficients(true);
        // 必要的配置，设置分割的模型类型，所用的随机参数估计方法，距离阀值，输入点云
        seg.setModelType(pcl::SACMODEL_NORMAL_PLANE);      // 设置模型类型
        seg.setMethodType(pcl::SAC_RANSAC);                // 设置随机采样一致性方法类型
        seg.setNormalDistanceWeight(_normal_dist_weight);            //设置表面法线权重系数
        seg.setMaxIterations(_fliter_max_iteration);
        seg.setDistanceThreshold(_distance_threshold);
        seg.setInputCloud(cloud);
        seg.setInputNormals(cloud_normals);
        //引发分割实现，存储分割结果到点几何inliers及存储平面模型的系数coefficients
        seg.segment(*inliers, *coefficients);
        if (inliers->indices.empty())
        {
            cout<<"error! Could not found any inliers!"<<endl;
        }
        // extract ground
        // 从点云中抽取分割的处在平面上的点集
        pcl::ExtractIndices<PointType> extractor;//点提取对象
        extractor.setInputCloud(cloud);
        extractor.setIndices(inliers);
        extractor.setNegative(true);        // true 表示滤除地面 false表示提取地面
        extractor.filter(*cloud_remove);
//        pcl::PointCloud<PointType>::Ptr cloud_pass(new pcl::PointCloud<PointType>);

        // 对传感器高度范围内的点云再进行一次筛选
        pcl::PassThrough<PointType> pass;
        pass.setInputCloud(cloud_remove);
        pass.setFilterFieldName("z");
        pass.setFilterLimits (-_sensor_height, 3);
        pass.filter(*cloud_filtered);
//        cout << "filter done."<<endl;
        // 体素下采样过大的话，会有空洞点，对后续卷积是否会有影响？
        if(isvoxeled){
            pcl::VoxelGrid<PointType> vg; //体素栅格下采样对象
            vg.setInputCloud (cloud_filtered);
            vg.setLeafSize (0.08f, 0.08f, 0.08f); //设置采样的体素大小
            vg.filter (*cloud_filtered);  //执行采样保存数据
        }
        // 显示点云
        if(_show){


        }

    }else{
        cout<<"no raw PointCloud data!"<<endl;
    }

}
PointType extractsegments::calculate_centroid(pcl::PointCloud<PointType>::Ptr& cloud) {
    Eigen::Vector4f centroid;
    pcl::compute3DCentroid(*cloud, centroid);
    return {centroid(0), centroid(1), centroid(2)};
}

double extractsegments::calculate_area_of_triangle(const PointType& pointa, const PointType& pointb, const PointType& pointc) {
    double dista = pcl::euclideanDistance(pointa, pointb);
    double distb = pcl::euclideanDistance(pointa, pointc);
    double distc = pcl::euclideanDistance(pointc, pointb);
    double p = 0.5 * (dista + distb + distc);
    double res = 0;
    double halensum = p * (p - dista) * (p - distb) * (p - distc);
    if(halensum > 0)    res = sqrt(halensum);
    return res;
}

std::vector<double> extractsegments::calculate_spatial_area(const std::vector<pair<PointType, double>>& centroids, int topk){
    vector<double> areas;
    double area_mat[centroids.size()][topk-2];  // 面积缓存表
    double area_factor[centroids.size()][topk-2];
    for(int i = 0; i< centroids.size(); i++){
        vector<pair<PointType, double> > topkpoints;
        // 计算所有距离，选取ktop个最近邻
        for(int j = 0; j< centroids.size(); j++){
            double dist = pcl::euclideanDistance(centroids[i].first, centroids[j].first);
            topkpoints.emplace_back(make_pair(centroids[j].first, dist));
        }
        sort(topkpoints.begin(), topkpoints.end(), comparedepth);
        // 计算空间三角形面积以及每个三角形的权重因子
        PointType pointA = topkpoints[0].first, pointB = topkpoints[1].first;
        double mindist = topkpoints[2].second;
        double factor_sum = 0;
        for(int m = 2; m < topk; m++){
            area_mat[i][m] = calculate_area_of_triangle(pointA, pointB, topkpoints[m].first);
            area_factor[i][m] = exp((topkpoints[m].second / mindist) * area_mat[i][m]);
            factor_sum += area_factor[i][m];
        }
        double spatial_area = 0;
        for(int n = 2; n < topk; n++){
            spatial_area += (area_factor[i][n] / factor_sum) * area_mat[i][n];
        }
        //cout<<"'"<<i<<"':"<<spatial_area<<endl;
        areas.push_back(spatial_area);
        topkpoints.clear();
    }
    double maxarea = 0;
    for(const double &area : areas) maxarea = maxarea > area ? maxarea : area;
    for(double &area : areas)   area = area / maxarea;
    return areas;
}

bool extractsegments::comparedepth(const pair<PointType, double> &pointdepthA,
                                   const pair<PointType, double> &pointdepthB) {
    return pointdepthA.second < pointdepthB.second;
}

std::vector<pcl::PointCloud<PointType>::Ptr> extractsegments::extract_segments(const pcl::PointCloud<PointType>::Ptr &cloud_filtered) {
    std::vector<pcl::PointIndices> cluster_indices = extract_cluster_indices(cloud_filtered);

    vector<pcl::PointCloud<PointType>::Ptr> Eucluextra; // 用于储存欧式分割后的点云
    for(auto & cluster_indice : cluster_indices){
        pcl::PointCloud<PointType>::Ptr cloud_cluster(new pcl::PointCloud<PointType>);
        double maxpointx = 0, maxpointy = 0, maxpointz = 0;
        double minpointx = INT_MAX, minpointy = INT_MAX, minpointz = INT_MAX;
        for(int & indice : cluster_indice.indices){
            cloud_cluster->points.push_back(cloud_filtered->points[indice]);
            maxpointx = maxpointx > cloud_filtered->points[indice].x ? maxpointx : cloud_filtered->points[indice].x;
            maxpointy = maxpointy > cloud_filtered->points[indice].y ? maxpointy : cloud_filtered->points[indice].y;
            maxpointz = maxpointz > cloud_filtered->points[indice].z ? maxpointz : cloud_filtered->points[indice].z;
            minpointx = minpointx < cloud_filtered->points[indice].x ? minpointx : cloud_filtered->points[indice].x;
            minpointy = minpointy < cloud_filtered->points[indice].y ? minpointy : cloud_filtered->points[indice].y;
            minpointz = minpointz < cloud_filtered->points[indice].z ? minpointz : cloud_filtered->points[indice].z;
        }
        double x_diff = maxpointx - minpointx;
        double y_diff = maxpointy - minpointy;
        double z_diff = maxpointz - minpointz;
        if(!_filter_flat_seg || max(x_diff, y_diff) / z_diff < _horizontal_ratio){
            cloud_cluster->width = cloud_cluster->points.size();
            cloud_cluster->height = 1;
            cloud_cluster->is_dense = true;
            Eucluextra.push_back(cloud_cluster);
        }
    }
//    cout<<"segments size:"<<Eucluextra.size()<<endl;
    return Eucluextra;
}

void extractsegments::show_senmantic_points(const pcl::PointCloud<PointType>::Ptr& cloud2show,
                                            const std::vector<int>& labels){
    const pcl::PointCloud<pcl::PointXYZI>::Ptr xyzi_cloud(new pcl::PointCloud<pcl::PointXYZI>());
    for(const auto& point: cloud2show->points){
        pcl::PointXYZI this_point;
        this_point.x = point.x;
        this_point.y = point.y;
        this_point.z = point.z;
        this_point.intensity = point.intensity;
        xyzi_cloud->push_back(this_point);
    }
    boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer(new pcl::visualization::PCLVisualizer(" cloud viewer"));
    viewer->setBackgroundColor(0, 0, 0);
    viewer->addPointCloud(xyzi_cloud, "fliter cloud");
    viewer->addCoordinateSystem(1.0);
    viewer->spin();
}






