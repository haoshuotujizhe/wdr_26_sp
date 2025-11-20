#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <cmath>
#include "tasks/auto_aim/yolo.hpp"
#include "tasks/auto_aim/armor.hpp"

// 全局变量 - 从配置文件读取
cv::Mat K;                    // 相机内参矩阵
cv::Mat distCoeffs;           // 畸变系数
float armor_width;           // 装甲板宽度
float armor_height;          // 装甲板高度  
float armor_pitch_deg;       // 装甲板倾角（度）
cv::Point3f armor_center_3d; // 装甲板中心在靶车坐标系中的坐标
cv::Point3f opp_armor_center_3d; // 对面装甲板中心坐标
cv::Mat R_camera2gimbal;
cv::Mat t_camera2gimbal;
cv::Mat R_gimbal2imubody;
cv::Vec4f imu_q; // IMU四元数 (w, x, y, z)

// 装甲板角点结构体
struct ArmorCorners {
    cv::Point3f p1, p2, p3, p4; // 左上、右上、右下、左下
};

/**
 * @brief 从YAML配置文件加载参数
 */
bool loadConfig(const std::string& config_path) {
    cv::FileStorage fs(config_path, cv::FileStorage::READ);
    if (!fs.isOpened()) {
        std::cerr << "Failed to open config file: " << config_path << std::endl;
        return false;
    }

    // 读取相机内参
    cv::FileNode camera_node = fs["camera"];
    cv::Mat camera_matrix;
    camera_node["camera_matrix"] >> camera_matrix;
    camera_node["dist_coeffs"] >> distCoeffs;
    
    // 确保内参矩阵是3x3
    if (camera_matrix.rows == 3 && camera_matrix.cols == 3) {
        K = camera_matrix;
    } else {
        std::cerr << "Invalid camera matrix size" << std::endl;
        return false;
    }

    // 读取装甲板参数
    cv::FileNode armor_node = fs["armor"];
    armor_width = armor_node["width"];
    armor_height = armor_node["height"];
    armor_pitch_deg = armor_node["pitch_deg"];
    
    // 读取装甲板3D坐标
    cv::FileNode center_node = armor_node["center"];
    armor_center_3d.x = center_node["x"];
    armor_center_3d.y = center_node["y"]; 
    armor_center_3d.z = center_node["z"];
    
    cv::FileNode opp_center_node = armor_node["opp_center"];
    opp_armor_center_3d.x = opp_center_node["x"];
    opp_armor_center_3d.y = opp_center_node["y"];
    opp_armor_center_3d.z = opp_center_node["z"];

    //读取转换矩阵
    camera_node["R_camera2gimbal"] >> R_camera2gimbal;
    camera_node["t_camera2gimbal"] >> t_camera2gimbal;
    camera_node["R_gimbal2imubody"] >> R_gimbal2imubody;

    // 读取IMU四元数
    camera_node["imu_q"] >> imu_q;

    fs.release();
    
    std::cout << "Config loaded successfully!" << std::endl;
    std::cout << "Camera matrix: \n" << K << std::endl;
    std::cout << "Armor size: " << armor_width << " x " << armor_height << std::endl;
    
    return true;
}

// -------------------- 计算旋转矩阵 Rz * Ry ------------------------
cv::Mat computeRotationMatrix(float yaw, float pitch) {
    float cy = std::cos(yaw);
    float sy = std::sin(yaw);
    float cp = std::cos(pitch);
    float sp = std::sin(pitch);

    // Rz(yaw)
    cv::Mat Rz = (cv::Mat_<float>(3,3) <<
        cy, -sy, 0,
        sy,  cy, 0,
         0,   0, 1
    );

    // Ry(pitch)  
    cv::Mat Ry = (cv::Mat_<float>(3,3) <<
         cp, 0, sp,
          0, 1,  0,
        -sp, 0, cp
    );

    return Rz * Ry;
}

// -------------------- 计算装甲板四个角点在靶车轨道坐标系下 ------------------------
ArmorCorners computeArmorCorners(
    const cv::Point3f& centerArmor, 
    const cv::Point3f& centerOppArmor,
    float width,
    float height,
    float pitchDeg   // 用户给出的倾角（度）
) {
    // 1. 算 yaw（水平朝向角）
    float vx = centerOppArmor.x - centerArmor.x;
    float vy = centerOppArmor.y - centerArmor.y;
    float yaw = std::atan2(vy, vx);  // 弧度

    // 2. pitch 角度转弧度
    float pitch = pitchDeg * 3.14159265358979f / 180.0f;

    // 3. 构造旋转矩阵
    cv::Mat R = computeRotationMatrix(yaw, pitch);

    // 4. 装甲板局部角点（中心为原点）
    float hw = width * 0.5f;
    float hh = height * 0.5f;

    std::vector<cv::Point3f> local = {
        cv::Point3f(-hw,  hh, 0),  // 左上
        cv::Point3f( hw,  hh, 0),  // 右上  
        cv::Point3f( hw, -hh, 0),  // 右下
        cv::Point3f(-hw, -hh, 0)   // 左下
    };

    ArmorCorners out;
    cv::Point3f* outs[4] = {&out.p1, &out.p2, &out.p3, &out.p4};

    // 5. 应用旋转 + 偏移
    for (int i = 0; i < 4; i++) {
        cv::Mat pt = (cv::Mat_<float>(3,1) << local[i].x, local[i].y, local[i].z);
        cv::Mat world = R * pt;

        outs[i]->x = centerArmor.x + world.at<float>(0,0);
        outs[i]->y = centerArmor.y + world.at<float>(1,0); 
        outs[i]->z = centerArmor.z + world.at<float>(2,0);
    }

    return out;
}

/**
 * @brief 从图像中检测装甲板，返回四个角点的像素坐标
 */
bool detectArmorCorners(
    const cv::Mat& img,
    const std::string& config_path, 
    std::vector<std::vector<cv::Point2f>>& corners_list
) {
    // 创建YOLO检测器
    static auto_aim::YOLO detector(config_path, false);
    
    // 检测装甲板
    auto armors = detector.detect(img);
    
    if (armors.empty()) {
        std::cout << "No armor detected!" << std::endl;
        return false;
    }
    
    corners_list.clear();
    
    // 遍历所有检测到的装甲板
    for (const auto& armor : armors) {
        std::vector<cv::Point2f> corners;
        
        // 从armor.bbox提取四个角点
        cv::Point2f vertices[4];
        armor.bbox.points(vertices);
        
        // 按顺序：左上、右上、右下、左下
        for (int i = 0; i < 4; i++) {
            corners.push_back(vertices[i]);
        }
        
        corners_list.push_back(corners);
        
        // 打印装甲板信息
        std::cout << "Armor detected - ID: " << armor.id 
                  << ", Color: " << (armor.color == auto_aim::Color::blue ? "Blue" : "Red")
                  << std::endl;
    }
    
    return true;
}

/**
 * @brief 使用 solvePnP 计算靶车坐标系到相机坐标系的齐次变换矩阵
 */
bool solvePnP_TargetToCamera(
    const std::vector<cv::Point3f>& objectPoints,
    const std::vector<cv::Point2f>& imagePoints, 
    cv::Mat& T_cam_target
) {
    if (objectPoints.size() != imagePoints.size() || objectPoints.size() < 4) {
        std::cerr << "Error: solvePnP requires at least 4 corresponding points." << std::endl;
        return false;
    }

    // solvePnP 结果：旋转向量 rvec 和平移向量 tvec
    cv::Mat rvec, tvec;

    // 调用 solvePnP
    bool ok = cv::solvePnP(
        objectPoints,
        imagePoints, 
        K,            // 相机内参
        distCoeffs,   // 畸变系数
        rvec,
        tvec,
        false,
        cv::SOLVEPNP_ITERATIVE
    );

    if (!ok) {
        std::cerr << "SolvePnP failed! Could not find transformation." << std::endl;
        return false;
    }

    // rvec -> R (旋转矩阵)
    cv::Mat R;
    cv::Rodrigues(rvec, R);

    // 生成 4x4 齐次矩阵 T_cam_target (靶车坐标系 -> 相机坐标系)
    T_cam_target = cv::Mat::eye(4, 4, CV_64F);
    
    // 复制 R 到 3x3 旋转部分
    R.copyTo(T_cam_target(cv::Range(0, 3), cv::Range(0, 3)));
    
    // 复制 tvec 到 3x1 平移部分  
    tvec.copyTo(T_cam_target(cv::Range(0, 3), cv::Range(3, 4)));

    // 打印结果进行验证
    std::cout << "--- solvePnP Results ---" << std::endl;
    std::cout << "Rotation Matrix R = \n" << R << std::endl;
    std::cout << "Translation tvec = \n" << tvec << std::endl;
    std::cout << "T_cam_target (Target -> Camera) = \n" << T_cam_target << std::endl;
    std::cout << "------------------------" << std::endl;

    return true;
}


/**
 * @brief 从四元数转换为旋转矩阵
 * @param q 四元数
 * @return 旋转矩阵 3x3
 */
cv::Mat quaternionToRotationMatrix(const cv::Vec4f& q) {
    float w = q[0], x = q[1], y = q[2], z = q[3];

    cv::Mat R = (cv::Mat_<float>(3,3) <<
        1 - 2 * (y * y + z * z),  2 * (x * y - z * w),  2 * (x * z + y * w),
        2 * (x * y + z * w),  1 - 2 * (x * x + z * z),  2 * (y * z - x * w),
        2 * (x * z - y * w),  2 * (y * z + x * w),  1 - 2 * (x * x + y * y)
    );

    return R;
}

/**
 * @brief 计算靶车轨道坐标系与IMU绝对世界坐标系之间的变换
 * @param T_camera2track 相机到靶车轨道坐标系的变换矩阵（4x4）
 * @param R_camera2gimbal 相机到云台的旋转矩阵（3x3）
 * @param t_camera2gimbal 相机到云台的平移向量（3x1）
 * @param R_gimbal2imubody 云台到IMU的旋转矩阵（3x3）
 * @param imu_quaternion IMU坐标系到世界坐标系的四元数（w, x, y, z）
 * @param T_track2imu_world 输出的靶车轨道坐标系到IMU绝对世界坐标系的变换（4x4）
 * @return 是否成功
 */
bool calculateTrackToImuWorldTransform(
    const cv::Mat& T_camera2track,
    const cv::Mat& R_camera2gimbal,
    const cv::Mat& t_camera2gimbal,
    const cv::Mat& R_gimbal2imubody,
    const cv::Vec4f& imu_quaternion,  // 四元数 (w, x, y, z)
    cv::Mat& T_track2imu_world
) {
    // 1. 相机到云台变换（齐次矩阵）
    cv::Mat T_camera2gimbal = cv::Mat::eye(4, 4, CV_64F);
    R_camera2gimbal.copyTo(T_camera2gimbal(cv::Range(0, 3), cv::Range(0, 3)));
    t_camera2gimbal.copyTo(T_camera2gimbal(cv::Range(0, 3), cv::Range(3, 4)));

    // 2. 云台到IMU变换（齐次矩阵）
    cv::Mat T_gimbal2imu = cv::Mat::eye(4, 4, CV_64F);
    R_gimbal2imubody.copyTo(T_gimbal2imu(cv::Range(0, 3), cv::Range(0, 3)));

    // 3. IMU到世界变换（通过四元数得到旋转矩阵，再转为齐次矩阵）
    cv::Mat R_imu2world;
    quaternionToRotationMatrix(imu_quaternion).convertTo(R_imu2world, CV_64F);
    cv::Mat T_imu2world = cv::Mat::eye(4, 4, CV_64F);
    R_imu2world.copyTo(T_imu2world(cv::Range(0, 3), cv::Range(0, 3)));

    // 4. 组合变换：计算靶车轨道到IMU世界坐标系的变换
    cv::Mat T_track2camera = T_camera2track.inv();  // Track→Camera
    T_track2imu_world = T_imu2world * T_gimbal2imu * T_camera2gimbal * T_track2camera;

    return true;
}
/**
 * @brief 完整流程：检测装甲板并计算位姿变换
 */
bool processArmorDetection(
    const cv::Mat& image,
    const std::string& config_path,
    cv::Mat& T_cam_target,
    const cv::Vec4f& imu_q,
    cv::Mat& T_track2imu_world
) {
    // // 1. 加载配置（首次调用时加载）
    // static bool config_loaded = false;
    // if (!config_loaded) {
    //     if (!loadConfig(config_path)) {
    //         std::cerr << "Failed to load config!" << std::endl;
    //         return false;
    //     }
    //     config_loaded = true;
    // }

    // 2. 检测装甲板角点（像素坐标）
    std::vector<std::vector<cv::Point2f>> detected_corners_list;
    if (!detectArmorCorners(image, config_path, detected_corners_list)) {
        return false;
    }

    // 3. 计算装甲板在靶车坐标系中的3D角点
    ArmorCorners armor_3d_corners = computeArmorCorners(
        armor_center_3d, 
        opp_armor_center_3d,
        armor_width, 
        armor_height,
        armor_pitch_deg
    );

    // 转换为solvePnP需要的格式
    std::vector<cv::Point3f> object_points_3d = {
        armor_3d_corners.p1, armor_3d_corners.p2, 
        armor_3d_corners.p3, armor_3d_corners.p4
    };

    // 4. 使用第一个检测到的装甲板进行位姿估计
    std::vector<cv::Point2f> image_points_2d = detected_corners_list[0];

    // 5. 计算变换矩阵
    solvePnP_TargetToCamera(object_points_3d, image_points_2d, T_cam_target);

    // 6. 计算靶车轨道坐标系到IMU绝对世界坐标系的变换
    if (!calculateTrackToImuWorldTransform(
            T_cam_target,
            R_camera2gimbal,
            t_camera2gimbal,
            R_gimbal2imubody,
            imu_q,
            T_track2imu_world)) {
        return false;
    }

    return true;
}


// 主函数示例
int main(int argc, char** argv) {
    // 配置文件路径
    std::string config_path = "config.yaml";
    
    // 读取图像
    cv::Mat image = cv::imread("test_image.jpg");
    if (image.empty()) {
        std::cerr << "Failed to load image!" << std::endl;
        return -1;
    }
    // 加载配置
    loadConfig(config_path);

    // 处理装甲板检测和位姿估计
    cv::Mat T_cam_target;
    if (processArmorDetection(image, config_path, T_cam_target, imu_q, T_cam_target)) {
        std::cout << "Armor detection and pose estimation successful!" << std::endl;
        // 这里可以使用 T_cam_target 进行后续处理
    } else {
        std::cout << "Armor detection failed!" << std::endl;
    }

    return 0;
}