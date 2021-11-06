#include <iostream>
#include "extract_segments.h"
#include "projection.h"

using namespace std;
using namespace cv;

int main() {
    pcl::PCDReader reader;
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud (new pcl::PointCloud<pcl::PointXYZ>);
    reader.read ("/home/wy-lab/Documents/GitHub/Place_recognization/pcd/0.pcd", *cloud);
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
    string savepath = "/home/wy-lab/Documents/GitHub/Place_recognization/project_imgs/";
    string rangename = "RangMat";
    string depthordername = "DepthOrderMat";
    string spatialname = "SpatialAreaMat";
    string scancontextname = "ScanContextMat";
    string normalname = "NomalMat";
    vector<string> names = {rangename, depthordername, spatialname, normalname, scancontextname};

    if(isspatial){
        for(int i=0; i< 5; i++){
            imwrite(savepath + names[i] + ".jpg", projections[i]);
        }
    }
//    vector<Mat> projections(4);
//    if(isspatial){
//        for(int i=0; i< 4; i++){
//            projections[i] = imread(savepath + names[i] + ".tiff", CV_32FC1);
//            imshow(string(names[i]), projections[i]);
//        }
//    }
//    waitKey(0);
    return 0;
}
