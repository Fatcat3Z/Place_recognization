#include <iostream>
#include "extract_segments.h"
#include "projection.h"

using namespace std;

int main() {
    pcl::PCDReader reader;
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud (new pcl::PointCloud<pcl::PointXYZ>);
    reader.read ("/Users/fatcat/Desktop/graduate/SLAM_WorkSpace/Place_recognization/pcd/0.pcd", *cloud);
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_flitered (new pcl::PointCloud<pcl::PointXYZ>);

    extractsegments extractor;
    extractor.filtercloud(cloud, cloud_flitered, true);
    vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> Eucluextra = extractor.extract_segments(cloud_flitered);
    pcl::PointXYZ pointa = {0, 0, 0};
    pcl::PointXYZ pointb = {1, 0, 0};
    pcl::PointXYZ pointc = {0, 0, 1};
//    vector<pcl::PointXYZ> cluster = {pointa, pointb, pointc};
//    vector<double> area = extractor.calculate_spatial_area(cluster, 3);
    RangeProjection projection(false, true);
    projection.getprojection(Eucluextra);

    return 0;
}
