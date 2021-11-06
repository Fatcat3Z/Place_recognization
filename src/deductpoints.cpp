//
// Created by wy-lab on 2021/10/18.
//
#include "deductpoints.h"
#include <iostream>
#include <fstream>

using namespace std;

bool deductpts::isinbox(const std::vector<float> &boxparamters, const pcl::PointXYZ &checkpoint) {
    float angle = boxparamters[6];
    pcl::PointXYZ rotatepoint = rotatept(checkpoint, angle);
    float dx = boxparamters[3] * 1.05;
    float dy = boxparamters[4] * 1.05;
    float dz = boxparamters[5] * 1.05;
    float diffx = abs(rotatepoint.x - boxparamters[0]);
    float diffy = abs(rotatepoint.y - boxparamters[1]);
    float diffz = abs(rotatepoint.z - boxparamters[2]);
    if(diffx < dx && diffy < dy && diffz < dz)
        return true;
    else
        return false;
}

pcl::PointXYZ deductpts::rotatept(const pcl::PointXYZ& targetpt, const float& angle) {
    float cosvalue = cos(angle);
    float sinvalue = sin(angle);
    Eigen::Vector3f targetpyvec(targetpt.x, targetpt.y, targetpt.z);
    Eigen::Matrix3f rotatematrix = Eigen::Matrix3f::Identity();
    rotatematrix(0, 0) = cosvalue;
    rotatematrix(0, 1) = sinvalue;
    rotatematrix(1, 0) = -sinvalue;
    rotatematrix(1, 1) = cosvalue;

    Eigen::Vector3f rotatevec = rotatematrix * targetpyvec;
    return {rotatevec[0], rotatevec[1], rotatevec[2]};
}
int deductpts::findnearestbox(const vector<std::vector<float>> &boxparamters, const pcl::PointXYZ &checkpoint) {
    int boxnum = boxparamters.size();
    int nearestidx = 0;
    float mindifx = abs(boxparamters[0][0] - checkpoint.x);
    float mindify = abs(boxparamters[0][1] - checkpoint.y);
    for(int i=1; i< boxnum; i++){
        float dx = abs(boxparamters[i][0] - checkpoint.x);
        float dy = abs(boxparamters[i][1] - checkpoint.y);
        nearestidx = (dx + dy) < (mindifx + mindify) ? i : nearestidx;
        mindifx = (dx + dy) < (mindifx + mindify) ? dx : mindifx;
        mindify = (dx + dy) < (mindifx + mindify) ? dy : mindify;
    }
    return nearestidx;
}

pcl::PointCloud<pcl::PointXYZ>::Ptr deductpts::removeobjs(pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud, const std::vector<std::vector<float>>& boxes) {
    pcl::PointCloud<pcl::PointXYZ>::Ptr removedobjs(new pcl::PointCloud<pcl::PointXYZ>);
    if(boxes.empty())   return cloud;
    for(const auto& point : *cloud){
        int boxidx = findnearestbox(boxes, point);
        if(!isinbox(boxes[boxidx], point))
            (*removedobjs).push_back(point);
    }
    return removedobjs;
}

std::vector<std::vector<float>> deductpts::loadboxes(const string &boxespath) {
    ifstream infile;
    infile.open(boxespath.data());
    assert(infile.is_open());
    string s;
    vector<vector<float>>  boxes;
    while(getline(infile, s)){
        string element;
        vector<float> box;
        for(const auto& boxelement: s){
            if(boxelement == ','){
                box.push_back(atof(element.c_str()));
                element.clear();
            }else{
                element += boxelement;
            }
        }
        cout<<"\n";
        boxes.push_back(box);
    }
    return boxes;
}

Eigen::Quaternionf deductpts::getquaterion(const float& angle) {
    float cosvalue = cos(angle);
    float sinvalue = sin(angle);
    Eigen::Matrix3f rotatematrix = Eigen::Matrix3f::Identity();
    rotatematrix(0, 0) = cosvalue;
    rotatematrix(0, 1) = sinvalue;
    rotatematrix(1, 0) = -sinvalue;
    rotatematrix(1, 1) = cosvalue;
    Eigen::Quaternionf res(rotatematrix);
    return res;
}




