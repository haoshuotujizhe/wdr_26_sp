#include <opencv4/opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <cmath>
#include <string>
#include "../tasks/auto_aim/yolo.hpp"
#include "../tasks/auto_aim/armor.hpp"

/**
 * Final version - Armor pose estimation
 * - 将配置加载、几何建模、视觉检测、位姿求解、坐标系融合封装为类/方法
 * - 保留原有算法实现（solvePnP、Rz*Ry 旋转构造等）
 * - 做了清晰的注释、错误检查和返回值判断
 *
 * 说明：
 *  - 本文件为单文件示例，方便直接编译测试；工程化时建议拆分为 .h/.cpp
 *  - 假设 IMU 四元数格式为 (w, x, y, z)
 *  - 所有坐标系说明见注释（Target/Track/Camera/IMU/World）
 */

namespace armor_pose {

// --------------------------- Helper types ---------------------------
struct ArmorCorners3D {
    cv::Point3f lt, rt, rb, lb; // left-top, right-top, right-bottom, left-bottom
};

struct Config {
    cv::Mat camera_matrix;   // 3x3
    cv::Mat dist_coeffs;     // nx1

    float armor_width = 0.f;
    float armor_height = 0.f;
    float armor_pitch_deg = 0.f;

    cv::Point3f armor_center {0,0,0};
    cv::Point3f opp_armor_center {0,0,0};

    cv::Mat R_camera2gimbal;
    cv::Mat t_camera2gimbal;
    cv::Mat R_gimbal2imubody;

    cv::Vec4f imu_q {1.f, 0.f, 0.f, 0.f}; // (w,x,y,z)
};

// --------------------------- Config loader ---------------------------
class ConfigLoader {
public:
    static bool loadFromYaml(const std::string& path, Config &cfg) {
        cv::FileStorage fs(path, cv::FileStorage::READ);
        if (!fs.isOpened()) {
            std::cerr << "[ConfigLoader] Failed open: " << path << std::endl;
            return false;
        }

        cv::FileNode cam = fs["camera"];
        cam["camera_matrix"] >> cfg.camera_matrix;
        cam["dist_coeffs"] >> cfg.dist_coeffs;

        if (cfg.camera_matrix.empty() || cfg.camera_matrix.rows != 3 || cfg.camera_matrix.cols != 3) {
            std::cerr << "[ConfigLoader] Invalid camera matrix." << std::endl;
            return false;
        }

        cv::FileNode armor = fs["armor"];
        armor["width"] >> cfg.armor_width;
        armor["height"] >> cfg.armor_height;
        armor["pitch_deg"] >> cfg.armor_pitch_deg;

        cv::FileNode center = armor["center"];
        center["x"] >> cfg.armor_center.x;
        center["y"] >> cfg.armor_center.y;
        center["z"] >> cfg.armor_center.z;

        cv::FileNode opp = armor["opp_center"];
        opp["x"] >> cfg.opp_armor_center.x;
        opp["y"] >> cfg.opp_armor_center.y;
        opp["z"] >> cfg.opp_armor_center.z;

        cam["R_camera2gimbal"] >> cfg.R_camera2gimbal;
        cam["t_camera2gimbal"] >> cfg.t_camera2gimbal;
        cam["R_gimbal2imubody"] >> cfg.R_gimbal2imubody;
        cam["imu_q"] >> cfg.imu_q;

        fs.release();
        std::cout << "[ConfigLoader] Loaded config: " << path << std::endl;
        return true;
    }
};

// --------------------------- Pose Estimator ---------------------------
class PoseEstimator {
public:
    PoseEstimator() = default;
    explicit PoseEstimator(const Config& cfg) { setConfig(cfg); }

    void setConfig(const Config& cfg) { config_ = cfg; }

    // R = Rz(yaw) * Ry(pitch) (float precision as original)
    static cv::Mat buildRotationMatrix(float yaw, float pitch) {
        float cy = std::cos(yaw), sy = std::sin(yaw);
        float cp = std::cos(pitch), sp = std::sin(pitch);

        cv::Mat Rz = (cv::Mat_<float>(3,3) <<
            cy, -sy, 0,
            sy,  cy, 0,
             0,   0, 1);

        cv::Mat Ry = (cv::Mat_<float>(3,3) <<
             cp, 0, sp,
              0, 1,  0,
            -sp, 0, cp);

        return Rz * Ry;
    }

    // 计算装甲板在靶车（track/target）坐标系下的四角点（保持原算法）
    ArmorCorners3D computeArmorCorners3D() const {
        ArmorCorners3D out;
        float dx = config_.opp_armor_center.x - config_.armor_center.x;
        float dy = config_.opp_armor_center.y - config_.armor_center.y;
        float yaw = std::atan2(dy, dx); // 弧度
        float pitch = config_.armor_pitch_deg * static_cast<float>(CV_PI) / 180.0f;

        cv::Mat R = buildRotationMatrix(yaw, pitch);

        float hw = config_.armor_width * 0.5f;
        float hh = config_.armor_height * 0.5f;

        std::vector<cv::Point3f> local = {
            {-hw,  hh, 0}, { hw,  hh, 0}, { hw, -hh, 0}, {-hw, -hh, 0}
        };

        cv::Point3f* dst[4] = { &out.lt, &out.rt, &out.rb, &out.lb };
        for (int i = 0; i < 4; ++i) {
            cv::Mat p = (cv::Mat_<float>(3,1) << local[i].x, local[i].y, local[i].z);
            cv::Mat w = R * p;
            dst[i]->x = config_.armor_center.x + w.at<float>(0);
            dst[i]->y = config_.armor_center.y + w.at<float>(1);
            dst[i]->z = config_.armor_center.z + w.at<float>(2);
        }

        return out;
    }

    // 使用内置 YOLO 检测器，返回角点列表（每个 armor 的 4 个像素点）
    // bool detectArmorCorners2D(const cv::Mat& image, const std::string& yolo_cfg,
    //                           std::vector<std::vector<cv::Point2f>>& corners_out) const
    // {
    //     static auto_aim::YOLO detector(yolo_cfg, false);
    //     auto armors = detector.detect(image);
    //     if (armors.empty()) {
    //         std::cout << "[PoseEstimator] No armor detected" << std::endl;
    //         return false;
    //     }

    //     corners_out.clear();
    //     for (const auto &a : armors) {
    //         cv::Point2f pts[4];
    //         a.bbox.points(pts);
    //         std::vector<cv::Point2f> v(pts, pts + 4);
    //         corners_out.push_back(v);
    //     }
    //     return true;
    // }
    
        // 修复：使用 Armor.points 而不是 bbox
    bool detectArmorCorners2D(const cv::Mat& image, const std::string& yolo_cfg,
                              std::vector<std::vector<cv::Point2f>>& corners_out) const
    {
        static auto_aim::YOLO detector(yolo_cfg, false);
        auto armors = detector.detect(image);
        
        if (armors.empty()) {
            std::cout << "[PoseEstimator] No armor detected" << std::endl;
            return false;
        }

        corners_out.clear();
        for (const auto &a : armors) {
            // 修复：直接使用 Armor 的 points 成员
            if (!a.points.empty() && a.points.size() >= 4) {
                // 确保有至少4个角点
                std::vector<cv::Point2f> corners;
                
                // 如果恰好是4个点，直接使用
                if (a.points.size() == 4) {
                    corners = a.points;
                } else {
                    // 如果超过4个点，取前4个（或根据需要选择）
                    corners.assign(a.points.begin(), a.points.begin() + 4);
                }
                
                corners_out.push_back(corners);
                
                std::cout << "[PoseEstimator] Armor detected - ID: " << static_cast<int>(a.name)
                          << ", Color: " << auto_aim::COLORS[a.color]
                          << ", Points: " << corners.size() << std::endl;
            } else {
                std::cerr << "[PoseEstimator] Warning: Armor has insufficient points ("
                          << a.points.size() << ")" << std::endl;
            }
        }
        
        return !corners_out.empty();
    }

    // solvePnP: objectPoints (3D in track/target) -> imagePoints (2D)
    bool solvePnP(const std::vector<cv::Point3f>& object_points,
                  const std::vector<cv::Point2f>& image_points,
                  cv::Mat &T_cam_target_out) const
    {
        if (object_points.size() != image_points.size() || object_points.size() < 4) {
            std::cerr << "[PoseEstimator] solvePnP: need >=4 correspondences" << std::endl;
            return false;
        }

        cv::Mat rvec, tvec;
        bool ok = cv::solvePnP(object_points, image_points,
                               config_.camera_matrix, config_.dist_coeffs,
                               rvec, tvec, false, cv::SOLVEPNP_ITERATIVE);
        if (!ok) {
            std::cerr << "[PoseEstimator] solvePnP failed" << std::endl;
            return false;
        }

        cv::Mat R;
        cv::Rodrigues(rvec, R);

        T_cam_target_out = cv::Mat::eye(4, 4, CV_64F);
        R.convertTo(T_cam_target_out(cv::Range(0,3), cv::Range(0,3)), CV_64F);
        tvec.convertTo(T_cam_target_out(cv::Range(0,3), cv::Range(3,4)), CV_64F);

        // Debug print (optional)
        std::cout << "[PoseEstimator] solvePnP R =" << R << "tvec = " << tvec << std::endl;
        return true;
    }

    // 四元数 (w,x,y,z) -> 3x3 旋转矩阵
    static cv::Mat quaternionToRotationMatrix(const cv::Vec4f &q) {
        float w = q[0], x = q[1], y = q[2], z = q[3];
        cv::Mat R = (cv::Mat_<double>(3,3) <<
            1 - 2*(y*y + z*z),  2*(x*y - z*w),    2*(x*z + y*w),
            2*(x*y + z*w),      1 - 2*(x*x + z*z),2*(y*z - x*w),
            2*(x*z - y*w),      2*(y*z + x*w),    1 - 2*(x*x + y*y)
        );
        return R;
    }

    // 计算 Track(目标轨道/靶车) -> IMU World 变换（齐次 4x4）
    bool computeTrackToImuWorld(const cv::Mat &T_cam_target, cv::Mat &T_track2imu_world_out) const {
        if (T_cam_target.empty()) return false;

        // T_camera2gimbal
        cv::Mat T_cam2gimbal = cv::Mat::eye(4,4,CV_64F);
        if (!config_.R_camera2gimbal.empty())
            config_.R_camera2gimbal.convertTo(T_cam2gimbal(cv::Range(0,3), cv::Range(0,3)), CV_64F);
        if (!config_.t_camera2gimbal.empty())
            config_.t_camera2gimbal.convertTo(T_cam2gimbal(cv::Range(0,3), cv::Range(3,4)), CV_64F);

        // T_gimbal2imu
        cv::Mat T_gimbal2imu = cv::Mat::eye(4,4,CV_64F);
        if (!config_.R_gimbal2imubody.empty())
            config_.R_gimbal2imubody.convertTo(T_gimbal2imu(cv::Range(0,3), cv::Range(0,3)), CV_64F);

        // T_imu2world (由四元数)
        cv::Mat R_imu2world = quaternionToRotationMatrix(config_.imu_q);
        cv::Mat T_imu2world = cv::Mat::eye(4,4,CV_64F);
        R_imu2world.copyTo(T_imu2world(cv::Range(0,3), cv::Range(0,3)));

        // T_track2camera = inv(T_cam_target)  （因为 T_cam_target: target->camera）
        cv::Mat T_target2camera = T_cam_target.clone();
        cv::Mat T_camera2target = T_target2camera.inv();

        // 组合：T_track2imu_world = T_imu2world * T_gimbal2imu * T_camera2gimbal * T_track2camera
        // 注：保持原始乘法顺序
        T_track2imu_world_out = T_imu2world * T_gimbal2imu * T_cam2gimbal * T_camera2target;
        return true;
    }

    // 全流程：检测->3D构造->solvePnP->坐标融合
    bool process(const cv::Mat &image, const std::string &yolo_cfg,
                 cv::Mat &T_cam_target_out, cv::Mat &T_track2imu_world_out) const
    {
        std::vector<std::vector<cv::Point2f>> detected_corners;
        if (!detectArmorCorners2D(image, yolo_cfg, detected_corners)) {
            return false;
        }

        ArmorCorners3D corners3d = computeArmorCorners3D();
        std::vector<cv::Point3f> object_pts = {corners3d.lt, corners3d.rt, corners3d.rb, corners3d.lb};
        std::vector<cv::Point2f> image_pts = detected_corners[0];

        if (!solvePnP(object_pts, image_pts, T_cam_target_out)) return false;

        if (!computeTrackToImuWorld(T_cam_target_out, T_track2imu_world_out)) return false;

        return true;
    }

private:
    Config config_;
};

} // namespace armor_pose

// --------------------------- Main 示例 ---------------------------
int main(int argc, char** argv) {
    std::string cfg_path = "../config.yaml";
    std::string image_path = "../test_image.jpg";
    std::string yolo_cfg = "../yolo_config.yaml"; // 传入 YOLO 配置（与 detector 构造函数匹配）

    armor_pose::Config cfg;
    if (!armor_pose::ConfigLoader::loadFromYaml(cfg_path, cfg)) {
        std::cerr << "Failed to load config" << std::endl;
        return -1;
    }

    cv::Mat image = cv::imread(image_path);
    if (image.empty()) {
        std::cerr << "Failed to read image: " << image_path << std::endl;
        return -1;
    }

    armor_pose::PoseEstimator estimator(cfg);

    cv::Mat T_cam_target, T_track2imu_world;
    if (estimator.process(image, yolo_cfg, T_cam_target, T_track2imu_world)) {
        std::cout << "[MAIN] Armor detection & pose estimation succeeded" << std::endl;
        std::cout << "T_cam_target:" << T_cam_target << std::endl;
        std::cout << "T_track2imu_world:" << T_track2imu_world << std::endl;
    } else {
        std::cout << "[MAIN] Armor detection or pose estimation failed" << std::endl;
    }

    return 0;
}

/*
==================== Final notes ====================
- 我把所有原有核心计算（旋转构造、角点局部坐标、solvePnP、四元数→矩阵、变换链）保留并封装为类方法。
- 若你需要把此文件拆成头/源文件或集成进现有工程，我可以帮你做：
  - 1) 拆分为 PoseEstimator.h / PoseEstimator.cpp
  - 2) 提供 CMakeLists.txt
  - 3) 添加单元测试与可视化（在图像上画出投影角点与坐标轴）

*/
