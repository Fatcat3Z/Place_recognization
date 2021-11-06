#include <iostream>
#include "extract_segments.h"
#include "projection.h"
#include <sys/types.h>
#include <dirent.h>
#include "deductpoints.h"

using namespace std;
using namespace cv;
void getFiles(const string& path, vector<string>& filenames)
{
    DIR *pDir;
    struct dirent* ptr;
    if(!(pDir = opendir(path.c_str()))){
        cout<<"Folder doesn't Exist!"<<endl;
        return;
    }
    while((ptr = readdir(pDir))!=0) {
        if (strcmp(ptr->d_name, ".") != 0 && strcmp(ptr->d_name, "..") != 0){
//            string filename = ptr->d_name;
//            string addzerofile = string(filelen - filename.length(), '0') + filename;
            filenames.emplace_back(ptr->d_name);
        }
    }
//    for(string& file: filenames){
//        file += string(filelen - file.size(), '0');
//    }
    closedir(pDir);
}

void getFileswithroot(const string& path, vector<string>& filenames)
{
    DIR *pDir;
    struct dirent* ptr;
    if(!(pDir = opendir(path.c_str()))){
        cout<<"Folder doesn't Exist!"<<endl;
        return;
    }
    while((ptr = readdir(pDir))!=0) {
        if (strcmp(ptr->d_name, ".") != 0 && strcmp(ptr->d_name, "..") != 0){
            filenames.emplace_back(path + ptr->d_name);
        }
    }
    closedir(pDir);
}

void addzeros(vector<string>& filenames, const int& filelen){
    for(string& file: filenames){
        file = string(filelen - file.size(), '0') + file;
    }
}

int main() {
    string datapath = "/home/wy-lab/kitti_odometry_dataset/sequences/02/pcd/";
//    string boxespath = "/home/wy-lab/kitti_odometry_dataset/sequences/02/boxesfiles/";
    vector<string> filenames;
//    vector<string> boxesparameters;
    getFiles(datapath, filenames);
    addzeros(filenames, 8);
//    getFileswithroot(boxespath, boxesparameters);
    sort(filenames.begin(), filenames.end());
//    sort(boxesparameters.begin(), boxesparameters.end());
    string saverootpath = "/home/wy-lab/kitti_odometry_dataset/sequences/02/RawImgs/";

    RangeProjection projection;

//    projection.projectscene(datapath, filenames, boxesparameters, saverootpath, false);
    projection.projectsegments(datapath, filenames, saverootpath);

//    for(int i = 0; i< filenames.size(); i++){
//
//        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud (new pcl::PointCloud<pcl::PointXYZ>);
//        reader.read (datapath + filenames[i], *cloud);
//        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_flitered (new pcl::PointCloud<pcl::PointXYZ>);
//        extractsegments extractor;
//        extractor.filtercloud(cloud, cloud_flitered, false);
//        deductpts remover;
//        vector<vector<float>> boxes = remover.loadboxes(boxesparameters[i]);
//        pcl::PointCloud<pcl::PointXYZ>::Ptr removedpld = remover.removeobjs(cloud_flitered, boxes);
//    }
//    string fileindex = "50.pcd";
//    pcl::PCDReader reader;
//    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud (new pcl::PointCloud<pcl::PointXYZ>);
//    reader.read (datapath + fileindex, *cloud);
//    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_flitered (new pcl::PointCloud<pcl::PointXYZ>);
//    extractsegments extractor;
//    extractor.filtercloud(cloud, cloud_flitered, false);
////    vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> Eucluextra = extractor.extract_segments(cloud_flitered);
////    extractor.savesegmentsbin(Eucluextra, savesegments + fileindex);
//    deductpts remover;
//    vector<vector<float>> boxes = remover.loadboxes(boxesfile);
//    pcl::PointCloud<pcl::PointXYZ>::Ptr removedpld = remover.removeobjs(cloud_flitered, boxes);
//    cout<<"raw cloud:"<<(*cloud).size()<<endl;
//    cout<<"before removing:"<<(*cloud_flitered).size()<<endl;
//    cout<<"after removing:"<<(*removedpld).size()<<endl;
//    boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer(new pcl::visualization::PCLVisualizer(" cloud viewer"));
//    viewer->setBackgroundColor(0, 0, 0);
//    int i = 0;
//    for(const auto& box: boxes){
//
//        Eigen::Vector3f position(box[0], box[1], box[2]);
//        Eigen::Quaternionf quat = remover.getquaterion(box[6]);
////        viewer->addCube(position, quat, box[3], box[4], box[5], "BBOX"+ to_string(i));
//        i++;
//    }
//    viewer->addPointCloud(removedpld, "fliter cloud");
//    viewer->addCoordinateSystem(1.0);
//    viewer->spin();

    return 0;
}
