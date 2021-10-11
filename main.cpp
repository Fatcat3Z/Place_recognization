#include <iostream>
#include "extract_segments.h"
#include "projection.h"

using namespace std;
using namespace cv;

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
    RangeProjection projection(false, false);
    bool isspatial = true;
    vector<Mat> projections = projection.getprojection(Eucluextra, isspatial);
    string savepath = "/Users/fatcat/Desktop/graduate/SLAM_WorkSpace/Place_recognization/project_imgs/";
    string rangename = "RangMat";
    string depthordername = "DepthOrderMat";
    string spatialname = "SpatialAreaMat";
    string scancontextname = "ScanContextMat";
    string normalname = "NomalMat";
    vector<string> names = {rangename, depthordername, spatialname, scancontextname, normalname};
    cout<<projections[1]<<endl;
    if(isspatial){
        for(int i=0; i< 4; i++){
            imwrite(savepath + names[i] + ".tiff", projections[i]);
        }
    }
    return 0;
}
