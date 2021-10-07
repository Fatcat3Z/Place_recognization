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
    extractor.filtercloud(cloud, cloud_flitered);
    vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> Eucluextra = extractor.extract_segments(cloud_flitered);
    RangeProjection projection(false, true);
    projection.frontproject(Eucluextra);
    return 0;
}
