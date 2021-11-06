//
// Created by FATCAT.STARK on 2021/10/4.
//
#include "projection.h"
#include "deductpoints.h"
#include <ctime>

using namespace std;
using namespace cv;
using namespace pcl;

vector<Mat> RangeProjection::getprojection(vector<pcl::PointCloud<pcl::PointXYZ>::Ptr>& Eucluextra, bool isspatialed) {

    // 计算点云的各项参数
//    boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer(new pcl::visualization::PCLVisualizer(" cloud viewer"));
//    if(_showclouds){
//        viewer->setBackgroundColor(0, 0, 0);
//        viewer->addCoordinateSystem(1.0);
//    }
    int i = 0;
    vector<pair<pcl::PointCloud<pcl::PointXYZ>::Ptr, float>> cloudndepth;
    vector<pair<pcl::PointXYZ, float>> centroidsndepth;
    map<PointXYZ, vector<float>, map_compare> pointnorder;
    vector<vector<float>> cloud_segments;
    set_segment_num(Eucluextra.size());
//    cout<<"segments_num"<<segments_num;
    // 这里的点云段还没有按照距离远近进行排序
    for(auto& cloud: Eucluextra){
        pcl::PointXYZ centroid = extractsegments::calculate_centroid(cloud);
        float range = pcl::euclideanDistance(PointXYZ(0,0,0), centroid);
        cloudndepth.emplace_back(make_pair(cloud, range));
        centroidsndepth.emplace_back(make_pair(centroid, range));
        for(const auto& point: *cloud){
            float depth = pcl::euclideanDistance(PointXYZ(0,0,0), point);
            cloud_segments.push_back({point.x, point.y, point.z, depth});
            pointnorder.insert(pair<pcl::PointXYZ, vector<float>>(point, {0, 0}));
        }
//        string str_filename = "cloud_segmented" + to_string(i);
//        pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> color(cloud, random() % 255, random() % 255, random() % 255);
//        i++;
//        if(_showclouds) viewer->addPointCloud(cloud, color, str_filename);
    }
//    if(_showclouds) viewer->spin();
    // 对所有点云按照距离远近降序排列
    sort(cloud_segments.begin(), cloud_segments.end(), comparevecdep);
    // 将所有段的质心按照距离降序排列，计算最近邻面积
    sort(centroidsndepth.begin(), centroidsndepth.end(), comparecentroiddep);
    vector<float> spatial_areas = extractor.calculate_spatial_area(centroidsndepth, _topk);
    // 对所有的段按照质心距离远近降序排列,并赋予距离次序以及
    sort(cloudndepth.begin(), cloudndepth.end(), compareclouddep);
    float depth_order = 1.0;
    for(auto& cloud: cloudndepth){
        for(const auto& point: *(cloud.first)){
            pointnorder.find(point)->second[0] = depth_order;
            pointnorder.find(point)->second[1] = spatial_areas[depth_order-1];
        }
        depth_order++;
    }
    vector<Mat> res = frontprojection(cloud_segments, pointnorder, isspatialed);
    Mat scancontext_proj = scancontext(cloud_segments);
    res.push_back(scancontext_proj);
    // 已经按照距离由远到近进行段排序，使距离近的点云能够在投影时覆盖距离元的点云
//    cout<<"project points:"<<cloud_segments.size()<<endl;

    // cout<<depth_order_mat<<endl;
    if(_showprojectios){
        imshow("ScanContext", scancontext_proj);
        moveWindow("ScanContext", 0, 90);
        waitKey(0);
    }
    return res;
}

vector<Mat> RangeProjection::getprojection(pcl::PointCloud<pcl::PointXYZ>::Ptr objremoved){
    Mat range_mat = Mat::zeros(_proj_H, _proj_W, CV_8UC1);
    Mat normal_mat = Mat::zeros(_proj_H, _proj_W, CV_32FC(3));
    float fov_up = _fov_up / 180.0 * CV_PI;
    float fov_down = _fov_down / 180.0 * CV_PI;
    float fov = abs(fov_up) + abs(fov_down);

    Mat scancontext = Mat::zeros(_proj_H, _proj_W, CV_8UC1);
    float gap_ring = _max_range / _scan_ring;      // 距离分辨率
    float gap_sector = 360.0 / _scan_sector;       // 角度分辨率

    for(const auto& point: *objremoved){

        float depth = pcl::euclideanDistance(PointXYZ(0,0,0), point);
        //        cout<<"depth:"<<depth<<endl;
        if(depth > _min_range && depth < _max_range){
            // 计算偏航角和俯仰角
            float yaw = - atan2(point.y, point.x);
            float pitch = asin(point.z / depth);
            // cout<<"depth:"<<(depth / _max_range) * 255<<endl;
            float proj_x = 0.5 * (yaw / CV_PI + 1.0);
            float proj_y = 1.0 - (pitch + abs(fov_down)) / fov;
            proj_x *= _proj_W;
            proj_y *= _proj_H;
            // 确保投影的上下界，防止溢出
            proj_x = floor(proj_x);
            proj_y = floor(proj_y);
            proj_x = proj_x > _proj_W - 1 ? _proj_W - 1 : proj_x;
            proj_x = proj_x > 0 ? proj_x : 0;
            proj_y = proj_y > _proj_H - 1 ? _proj_H - 1 : proj_y;
            proj_y = proj_y > 0 ? proj_y : 0;

            normal_mat.at<Vec3f>(int(proj_y), int(proj_x))[0] = point.x;
            normal_mat.at<Vec3f>(int(proj_y), int(proj_x))[1] = point.y;
            normal_mat.at<Vec3f>(int(proj_y), int(proj_x))[2] = point.z;

            //            cout<<"pointindicesmap("<<proj_x<<","<<proj_y<<"):"<<normal_mat.at<Vec3f>(proj_y, proj_x)[0]<<endl;
            range_mat.at<char>(int(proj_y), int(proj_x)) = (_min_range / depth) * 255;

            float point_height = (point.z + _sensor_height) / _scan_top;
            float theta = 180 * (yaw / CV_PI + 1.0);
            float faraway = sqrt(point.x * point.x + point.y * point.y);
            //  整除角度分辨率，获得编码序号
            int idx_ring = (int)(faraway / gap_ring);   // 图像纵坐标
            int idx_sector = (int)(theta / gap_sector); // 图像横坐标
            idx_ring = idx_ring >= _scan_ring ? _scan_ring - 1 : idx_ring;
            if(point_height > 0 && point_height < _scan_top){
                if(point_height > scancontext.at<float>(idx_ring, idx_sector)){
                    scancontext.at<char>(idx_ring, idx_sector) = point_height * 255;
                }
            }
        }
    }
    Mat normal_map = getnormalmap(normal_mat);
    return {range_mat, normal_map, scancontext};
}

std::vector<cv::Mat> RangeProjection::frontprojection(const std::vector<std::vector<float>>& cloud_segments,
                                         map<PointXYZ, vector<float>, map_compare> pointnorder,
                                         int state){
    Mat range_mat = Mat::zeros(_proj_H, _proj_W, CV_8UC1);
    Mat spatial_area_mat = Mat::zeros(_proj_H, _proj_W, CV_8UC1);
//    Mat depth_order_mat = Mat::zeros(_proj_H, _proj_W, CV_8UC1);
    Mat normal_mat = Mat::zeros(_proj_H, _proj_W, CV_32FC(3));
    // 雷达参数设置,上下视角转弧度制
    float fov_up = _fov_up / 180.0 * CV_PI;
    float fov_down = _fov_down / 180.0 * CV_PI;
    float fov = abs(fov_up) + abs(fov_down);

    for(const auto& point: cloud_segments){
        pcl::PointXYZ targetpoint(point[0], point[1], point[2]);
        float depth = point[3];
//        cout<<"depth:"<<depth<<endl;
        if(depth > _min_range && depth < _max_range){
            // 计算偏航角和俯仰角
            float yaw = - atan2(point[1], point[0]);
            float pitch = asin(point[2] / depth);
            // cout<<"depth:"<<(depth / _max_range) * 255<<endl;
            float proj_x = 0.5 * (yaw / CV_PI + 1.0);
            float proj_y = 1.0 - (pitch + abs(fov_down)) / fov;
            proj_x *= _proj_W;
            proj_y *= _proj_H;
            // 确保投影的上下界，防止溢出
            proj_x = floor(proj_x);
            proj_y = floor(proj_y);
            proj_x = proj_x > _proj_W - 1 ? _proj_W - 1 : proj_x;
            proj_x = proj_x > 0 ? proj_x : 0;
            proj_y = proj_y > _proj_H - 1 ? _proj_H - 1 : proj_y;
            proj_y = proj_y > 0 ? proj_y : 0;
            normal_mat.at<Vec3f>(int(proj_y), int(proj_x))[0] = point[0];
            normal_mat.at<Vec3f>(int(proj_y), int(proj_x))[1] = point[1];
            normal_mat.at<Vec3f>(int(proj_y), int(proj_x))[2] = point[2];
//            cout<<"pointindicesmap("<<proj_x<<","<<proj_y<<"):"<<normal_mat.at<Vec3f>(proj_y, proj_x)[0]<<endl;
            range_mat.at<char>(int(proj_y), int(proj_x)) = (_min_range / depth) * 255;
//            depth_order_mat.at<char>((int)proj_y, (int)proj_x) = (pointnorder.find(targetpoint)->second[0] / _segments) * 255;
            spatial_area_mat.at<char>((int)proj_y, (int)proj_x) = (pointnorder.find(targetpoint)->second[1]) * 255;
        }
    }
    Mat normal_map = getnormalmap(normal_mat);
    if(_showprojectios){
        imshow("spatial area projection", spatial_area_mat);
        moveWindow("spatial area projection", 0, 180);
        imshow("range projection", range_mat);
        moveWindow("range projection", 0, 270);
//        imshow("depth order projection", depth_order_mat);
//        moveWindow("depth order projection", 0, 360);
        imshow("normal projection", normal_map);
        moveWindow("normal projection", 0, 450);

    }
    if(state){
//        Mat spatial_projection;
//        vector<Mat> final(3);
//        final[0] = range_mat;
//        final[1] = depth_order_mat;
//        final[2] = spatial_area_mat;
//        merge(final, spatial_projection);
//        if(_showprojectios){
//            imshow("spatial_area projection", spatial_projection);
//            moveWindow("spatial_area projection", 0, 0);
//        }
        return {range_mat, spatial_area_mat, normal_map};
    }else{
        return {range_mat, normal_map};
    }
}

bool RangeProjection::comparevecdep(const std::vector<float>& cloudA,
                                    const std::vector<float>& cloudB) {
    return cloudA[3] > cloudB[3];
}

bool RangeProjection::compareclouddep(const pair<pcl::PointCloud<pcl::PointXYZ>::Ptr, float> &cloudA,
                                      const pair<pcl::PointCloud<pcl::PointXYZ>::Ptr, float> &cloudB) {
    return cloudA.second > cloudB.second;
}

bool RangeProjection::comparecentroiddep(const pair<pcl::PointXYZ, float> &cloudA,
                                    const pair<pcl::PointXYZ, float> &cloudB) {
    return cloudA.second > cloudB.second;
}

cv::Mat RangeProjection::scancontext(const std::vector<std::vector<float>>& cloud_segments) {
    Mat scancontext = Mat::zeros(_proj_H, _proj_W, CV_8UC1);
    float gap_ring = _max_range / _scan_ring;      // 距离分辨率
    float gap_sector = 360.0 / _scan_sector;       // 角度分辨率
    for(const auto& point: cloud_segments){
        float point_height = (point[2] + _sensor_height) / _scan_top;
        float yaw = - atan2(point[1], point[0]);
        float theta = 180 * (yaw / CV_PI + 1.0);
        float faraway = sqrt(point[0] * point[0] + point[1] * point[1]);
        //  整除角度分辨率，获得编码序号
        int idx_ring = (int)(faraway / gap_ring);   // 图像纵坐标
        int idx_sector = (int)(theta / gap_sector); // 图像横坐标
        idx_ring = idx_ring >= _scan_ring ? _scan_ring - 1 : idx_ring;
        if(point_height > 0 && point_height < _scan_top){
            if(point_height > scancontext.at<float>(idx_ring, idx_sector)){
                scancontext.at<char>(idx_ring, idx_sector) = point_height * 255;
            }
        }
    }
    return scancontext;
}

cv::Mat RangeProjection::getnormalmap(const cv::Mat& pointindicesmap) {

    Mat normal_mat = Mat::zeros(_proj_H, _proj_W, CV_8UC(3));
    for(int i = 0; i< _proj_H; i++){
        for(int j = 0; j< _proj_W; j++){
            if(pointindicesmap.at<Vec3f>(i, j)[0] != 0){
                int rightpointx = j+1 > _proj_W ? _proj_W : j+1;
                int downpointy = i+1 > _proj_H ? _proj_H : i+1;
                Eigen::Vector3f horvec(pointindicesmap.at<Vec3f>(i, j)[0] - pointindicesmap.at<Vec3f>(i, rightpointx)[0],
                                       pointindicesmap.at<Vec3f>(i, j)[1] - pointindicesmap.at<Vec3f>(i, rightpointx)[1],
                                       pointindicesmap.at<Vec3f>(i, j)[2] - pointindicesmap.at<Vec3f>(i, rightpointx)[2]);
                //                cout<<"horvec"<<horvec<<endl;
                //                horvec.normalize();
                Eigen::Vector3f vervec(pointindicesmap.at<Vec3f>(i, j)[0] - pointindicesmap.at<Vec3f>(downpointy, j)[0],
                                       pointindicesmap.at<Vec3f>(i, j)[1] - pointindicesmap.at<Vec3f>(downpointy, j)[1],
                                       pointindicesmap.at<Vec3f>(i, j)[2] - pointindicesmap.at<Vec3f>(downpointy, j)[2]);
                //                vervec.normalize();
                Eigen::Vector3f normal = horvec.cross(vervec);
                normal.normalize();

                normal_mat.at<Vec3b>(i, j)[0] = (char)(normal[0] + 1) * 127;
                normal_mat.at<Vec3b>(i, j)[1] = (char)(normal[1] + 1) * 127;
                normal_mat.at<Vec3b>(i, j)[2] = (char)(normal[2] + 1) * 127;

            }
        }
    }
    return normal_mat;
}

void RangeProjection::projectsegments(const std::string& rootpath, const std::vector<std::string>& filenames, const std::string& saverootpath) {
    float all_time = 0;
    float average_time = 0;
    int count = 1;
    for(const string& file: filenames){
            clock_t startTime,endTime;
            startTime = clock();
            pcl::PCDReader reader;
            pcl::PointCloud<pcl::PointXYZ>::Ptr cloud (new pcl::PointCloud<pcl::PointXYZ>);
            reader.read (rootpath + file, *cloud);
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
            endTime = clock();
            all_time += (float)(endTime - startTime) / CLOCKS_PER_SEC;
            average_time = all_time / count;
            cout << "The average run time is: " << average_time << "s" << endl;
            count++;
            if(isspatial){
                int pos=file.find_last_of('/');
                string s1(file.substr(pos + 1));
                int want_length = 4;
                string pcdnumber = s1.substr(0, s1.rfind('.'));
                string addzero(want_length - pcdnumber.length(), '0');
                imwrite(saverootpath + "SegRangeMat/" + addzero + pcdnumber + ".jpg", projections[0]);
                imwrite(saverootpath + "SegSpatialAreaMat/"+ addzero + pcdnumber + ".jpg", projections[1]);
                imwrite(saverootpath + "SegNormalMat/"+ addzero + pcdnumber + ".jpg", projections[2]);
                imwrite(saverootpath + "SegScanContextMat/"+ addzero + pcdnumber + ".jpg", projections[3]);
                cout<<"finish round"<<addzero + pcdnumber<<endl;
            }
        }
}

void RangeProjection::projectscene(const std::string& rootpath,
                                   const std::vector<std::string> &filenames,
                                   const std::vector<std::string>& boxfilepath,
                                   const std::string &saverootpath,
                                   bool segments) {
    float all_time = 0;
    float average_time = 0;
    int count = 1;
    for(int i = 0; i< filenames.size(); i++){
        clock_t startTime,endTime;
        startTime = clock();
        pcl::PCDReader reader;
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud (new pcl::PointCloud<pcl::PointXYZ>);
        reader.read (rootpath + filenames[i], *cloud);
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_flitered (new pcl::PointCloud<pcl::PointXYZ>);

        extractsegments extractor;
        extractor.filtercloud(cloud, cloud_flitered, true);
        deductpts remover;
        vector<vector<float>> boxes = remover.loadboxes(boxfilepath[i]);
        pcl::PointCloud<pcl::PointXYZ>::Ptr removedpld = remover.removeobjs(cloud_flitered, boxes);
        vector<Mat> projections;
        if(segments){
            vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> Eucluextra = extractor.extract_segments(cloud_flitered);
            bool isspatial = true;
            projections = getprojection(Eucluextra, isspatial);
        }else{
            projections = getprojection(removedpld);
        }

        endTime = clock();
        all_time += (float)(endTime - startTime) / CLOCKS_PER_SEC;
        average_time = all_time / count;
        cout << "The average run time is: " << average_time << "s" << endl;
        count++;
        if(segments){
            string savename = filenames[i].substr(0, 4);
            imwrite(saverootpath + "RmvSegRangeMat/" + savename + ".jpg", projections[0]);
            imwrite(saverootpath + "RmvSegDepthOrderMat/"+ savename + ".jpg", projections[1]);
            imwrite(saverootpath + "RmvSegSpatialAreaMat/"+ savename + ".jpg", projections[2]);
            imwrite(saverootpath + "RmvSegScanContextMat/"+ savename + ".jpg", projections[3]);
            imwrite(saverootpath + "RmvSegNormalMat/"+ savename + ".jpg", projections[4]);
            cout<<"finish round"<<savename<<endl;
        }else{
            string savename = filenames[i].substr(0, 4);
            imwrite(saverootpath + "RmvRangeMat/" + savename + ".jpg", projections[0]);
            imwrite(saverootpath + "RmvNormalMat/"+ savename + ".jpg", projections[1]);
            imwrite(saverootpath + "RmvScanContextMat/"+ savename + ".jpg", projections[2]);
            cout<<"finish round"<<savename<<endl;
        }
    }
}



