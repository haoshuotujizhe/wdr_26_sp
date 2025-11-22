// adaptive_transform.cpp
// 完全适配你的 config.yaml 的版本

#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <cmath>

using namespace std;

// ================= 工具函数 =================

cv::Mat quatToRot(const cv::Vec4d &q) {
    double w=q[0], x=q[1], y=q[2], z=q[3];
    double n = sqrt(w*w + x*x + y*y + z*z);
    if(n < 1e-6) return cv::Mat::eye(3,3,CV_64F);
    w/=n; x/=n; y/=n; z/=n;

    return (cv::Mat_<double>(3,3) <<
        1-2*(y*y+z*z), 2*(x*y - z*w), 2*(x*z + y*w),
        2*(x*y + z*w), 1-2*(x*x+z*z), 2*(y*z - x*w),
        2*(x*z - y*w), 2*(y*z + x*w), 1-2*(x*x+y*y)
    );
}

vector<cv::Point3f> makeArmorCorners(double w, double h) {
    double hw=w/2, hh=h/2;
    return {
        {-hw, hh, 0}, {hw, hh, 0},
        {hw, -hh, 0}, {-hw, -hh, 0}
    };
}

// ================= 主程序 =================

int main(int argc,char** argv){
    string cfg = "../config.yaml";
    if(argc>1) cfg = argv[1];

    cv::FileStorage fs(cfg, cv::FileStorage::READ);
    if(!fs.isOpened()){
        cout<<"Failed to open config"<<endl;
        return -1;
    }

    // -------- 读取相机参数 --------
    cv::Mat K, dist;
    fs["camera"]["camera_matrix"] >> K;
    fs["camera"]["dist_coeffs"] >> dist;

    cv::Mat R_cam2gimbal, t_cam2gimbal;
    fs["camera"]["R_camera2gimbal"] >> R_cam2gimbal;
    fs["camera"]["t_camera2gimbal"] >> t_cam2gimbal;

    cv::Mat R_gimbal2imu;
    fs["camera"]["R_gimbal2imubody"] >> R_gimbal2imu;

    // -------- IMU 四元数 --------
    cv::FileNode qnode = fs["imu_q"];
    cv::Vec4d imu_q;
    for(int i=0;i<4;i++) imu_q[i] = (double)qnode[i];
    cv::Mat R_world2imu = quatToRot(imu_q);

    // -------- 装甲板参数 --------
    double width, height, pitch_deg;
    fs["armor"]["width"]>>width;
    fs["armor"]["height"]>>height;
    fs["armor"]["pitch_deg"]>>pitch_deg;

    cv::Point3d center, opp;
    fs["armor"]["center"]["x"]>>center.x;
    fs["armor"]["center"]["y"]>>center.y;
    fs["armor"]["center"]["z"]>>center.z;
    fs["armor"]["opp_center"]["x"]>>opp.x;
    fs["armor"]["opp_center"]["y"]>>opp.y;
    fs["armor"]["opp_center"]["z"]>>opp.z;

    // -------- 图像角点 --------
    vector<cv::Point2f> img_pts;
    for(auto it: fs["image_corners"]) {
        float x,y;
        it["x"]>>x; it["y"]>>y;
        img_pts.emplace_back(x,y);
    }

    // ================= 生成装甲板 3D 点 =================
    vector<cv::Point3f> obj_pts;
    auto local=makeArmorCorners(width,height);

    double yaw = atan2(opp.y-center.y, opp.x-center.x);
    double pitch = pitch_deg * CV_PI / 180.0;

    cv::Mat Rz=(cv::Mat_<double>(3,3)<<cos(yaw),-sin(yaw),0,sin(yaw),cos(yaw),0,0,0,1);
    cv::Mat Ry=(cv::Mat_<double>(3,3)<<cos(pitch),0,sin(pitch),0,1,0,-sin(pitch),0,cos(pitch));
    cv::Mat R_armor = Rz * Ry;

    for(auto &p:local){
        cv::Mat v=(cv::Mat_<double>(3,1)<<p.x,p.y,p.z);
        cv::Mat w=R_armor*v;
        obj_pts.emplace_back(
            center.x + w.at<double>(0),
            center.y + w.at<double>(1),
            center.z + w.at<double>(2)
        );
    }

    // ================= solvePnP =================
    cv::Mat rvec,tvec;
    cv::solvePnP(obj_pts,img_pts,K,dist,rvec,tvec);

    cv::Mat R_target2cam;
    cv::Rodrigues(rvec,R_target2cam);

    // Track -> Camera
    cv::Mat T_target2cam=cv::Mat::eye(4,4,CV_64F);
    R_target2cam.copyTo(T_target2cam(cv::Range(0,3),cv::Range(0,3)));
    tvec.copyTo(T_target2cam(cv::Range(0,3),cv::Range(3,4)));

    // Camera -> IMU
    cv::Mat R_cam2imu = R_gimbal2imu * R_cam2gimbal;
    cv::Mat t_cam2imu = R_gimbal2imu * t_cam2gimbal;

    cv::Mat T_cam2imu=cv::Mat::eye(4,4,CV_64F);
    R_cam2imu.copyTo(T_cam2imu(cv::Range(0,3),cv::Range(0,3)));
    t_cam2imu.copyTo(T_cam2imu(cv::Range(0,3),cv::Range(3,4)));

    // Track -> IMU
    cv::Mat T_target2imu = T_cam2imu * T_target2cam;

    // IMU -> Track
    cv::Mat T_imu2target = T_target2imu.inv();

    cout<<"========= 结果 ========="<<endl;
    // cout<<"T_target2cam:\n"<<T_target2cam<<endl;
    cout<<"T_target2imu:\n"<<T_target2imu<<endl;
    // cout<<"T_imu2target:\n"<<T_imu2target<<endl;

    return 0;
}
