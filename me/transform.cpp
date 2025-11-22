#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <cmath>
#include <string>
#include <fstream>

namespace armor_pose {

// --------------------------- Helper types ---------------------------
struct ArmorCorners3D {
    cv::Point3f lt, rt, rb, lb;
};

struct Config {
    cv::Mat camera_matrix;
    cv::Mat dist_coeffs;

    float armor_width = 0.f;
    float armor_height = 0.f;
    float armor_pitch_deg = 0.f;

    cv::Point3f armor_center {0,0,0};
    cv::Point3f opp_armor_center {0,0,0};

    cv::Mat R_camera2gimbal;
    cv::Mat t_camera2gimbal;
    cv::Mat R_gimbal2imubody;

    cv::Vec4f imu_q {1.f, 0.f, 0.f, 0.f};

    // ✅ 直接从配置文件读取的像素坐标
    std::vector<cv::Point2f> image_corners;  // 左上、右上、右下、左下
};

// --------------------------- Config loader ---------------------------
class ConfigLoader {
public:
    static bool loadFromYaml(const std::string& path, Config &cfg) {
        std::ifstream test_file(path);
        if (!test_file.good()) {
            std::cerr << "[ConfigLoader] File not found: " << path << std::endl;
            return false;
        }
        test_file.close();

        cv::FileStorage fs;
        try {
            fs.open(path, cv::FileStorage::READ);
        } catch (const cv::Exception& e) {
            std::cerr << "[ConfigLoader] OpenCV exception: " << e.what() << std::endl;
            return false;
        }

        if (!fs.isOpened()) {
            std::cerr << "[ConfigLoader] Failed to open: " << path << std::endl;
            return false;
        }

        // ===== 读取相机参数 =====
        cv::FileNode cam = fs["camera"];
        if (cam.empty()) {
            std::cerr << "[ConfigLoader] Missing 'camera' node" << std::endl;
            fs.release();
            return false;
        }

        try {
            cam["camera_matrix"] >> cfg.camera_matrix;
            cam["dist_coeffs"] >> cfg.dist_coeffs;

            if (cfg.camera_matrix.empty() || cfg.camera_matrix.rows != 3) {
                std::cerr << "[ConfigLoader] Invalid camera_matrix" << std::endl;
                fs.release();
                return false;
            }
            std::cout << "[ConfigLoader] ✓ Camera matrix loaded" << std::endl;

            if (!cam["R_camera2gimbal"].empty()) {
                cam["R_camera2gimbal"] >> cfg.R_camera2gimbal;
                std::cout << "[ConfigLoader] ✓ R_camera2gimbal loaded" << std::endl;
            }

            if (!cam["t_camera2gimbal"].empty()) {
                cam["t_camera2gimbal"] >> cfg.t_camera2gimbal;
                std::cout << "[ConfigLoader] ✓ t_camera2gimbal loaded" << std::endl;
            }

            if (!cam["R_gimbal2imubody"].empty()) {
                cam["R_gimbal2imubody"] >> cfg.R_gimbal2imubody;
                std::cout << "[ConfigLoader] ✓ R_gimbal2imubody loaded" << std::endl;
            }

            cv::FileNode imu_q_node = cam["imu_q"];
            if (!imu_q_node.empty() && imu_q_node.isSeq()) {
                std::vector<float> imu_q_vec;
                imu_q_node >> imu_q_vec;
                
                if (imu_q_vec.size() == 4) {
                    cfg.imu_q = cv::Vec4f(imu_q_vec[0], imu_q_vec[1], 
                                          imu_q_vec[2], imu_q_vec[3]);
                    std::cout << "[ConfigLoader] ✓ imu_q: [" << cfg.imu_q[0] << ", " 
                              << cfg.imu_q[1] << ", " << cfg.imu_q[2] << ", " 
                              << cfg.imu_q[3] << "]" << std::endl;
                }
            }

        } catch (const cv::Exception& e) {
            std::cerr << "[ConfigLoader] Error reading camera params: " << e.what() << std::endl;
            fs.release();
            return false;
        }

        // ===== 读取装甲板参数 =====
        cv::FileNode armor = fs["armor"];
        if (!armor.empty()) {
            try {
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

                std::cout << "[ConfigLoader] ✓ Armor params: " 
                          << cfg.armor_width << "x" << cfg.armor_height << " m" << std::endl;
            } catch (const cv::Exception& e) {
                std::cerr << "[ConfigLoader] Error reading armor params: " << e.what() << std::endl;
            }
        }

        // ===== ✅ 读取图像角点坐标 =====
        cv::FileNode corners = fs["image_corners"];
        if (!corners.empty() && corners.isSeq()) {
            cfg.image_corners.clear();
            
            for (auto it = corners.begin(); it != corners.end(); ++it) {
                cv::FileNode pt = *it;
                float x = (float)pt["x"];
                float y = (float)pt["y"];
                cfg.image_corners.push_back(cv::Point2f(x, y));
            }
            
            if (cfg.image_corners.size() == 4) {
                std::cout << "[ConfigLoader] ✓ Image corners loaded:" << std::endl;
                std::cout << "  Left-Top:     (" << cfg.image_corners[0].x << ", " 
                          << cfg.image_corners[0].y << ")" << std::endl;
                std::cout << "  Right-Top:    (" << cfg.image_corners[1].x << ", " 
                          << cfg.image_corners[1].y << ")" << std::endl;
                std::cout << "  Right-Bottom: (" << cfg.image_corners[2].x << ", " 
                          << cfg.image_corners[2].y << ")" << std::endl;
                std::cout << "  Left-Bottom:  (" << cfg.image_corners[3].x << ", " 
                          << cfg.image_corners[3].y << ")" << std::endl;
            } else {
                std::cerr << "[ConfigLoader] ✗ Expected 4 corners, got " 
                          << cfg.image_corners.size() << std::endl;
                fs.release();
                return false;
            }
        } else {
            std::cerr << "[ConfigLoader] ✗ Missing 'image_corners' node" << std::endl;
            fs.release();
            return false;
        }

        fs.release();
        std::cout << "[ConfigLoader] ✓ Config loaded successfully" << std::endl;
        return true;
    }
};

// --------------------------- Pose Estimator ---------------------------
class PoseEstimator {
public:
    PoseEstimator() = default;
    explicit PoseEstimator(const Config& cfg) { setConfig(cfg); }

    void setConfig(const Config& cfg) { config_ = cfg; }

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

    ArmorCorners3D computeArmorCorners3D() const {
        ArmorCorners3D out;
        float dx = config_.opp_armor_center.x - config_.armor_center.x;
        float dy = config_.opp_armor_center.y - config_.armor_center.y;
        float yaw = std::atan2(dy, dx);
        float pitch = config_.armor_pitch_deg * static_cast<float>(M_PI) / 180.0f;

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

    bool solvePnP(const std::vector<cv::Point3f>& object_points,
                  const std::vector<cv::Point2f>& image_points,
                  cv::Mat &T_cam_target_out) const
    {
        if (object_points.size() != image_points.size() || object_points.size() < 4) {
            std::cerr << "[PoseEstimator] Invalid point count" << std::endl;
            return false;
        }

        std::cout << "[PoseEstimator] Solving PnP with " << object_points.size() 
                  << " point correspondences..." << std::endl;

        cv::Mat rvec, tvec;
        bool ok = cv::solvePnP(object_points, image_points,
                               config_.camera_matrix, config_.dist_coeffs,
                               rvec, tvec, false, cv::SOLVEPNP_ITERATIVE);
        if (!ok) {
            std::cerr << "[PoseEstimator] ✗ solvePnP failed" << std::endl;
            return false;
        }

        cv::Mat R;
        cv::Rodrigues(rvec, R);

        T_cam_target_out = cv::Mat::eye(4, 4, CV_64F);
        R.convertTo(T_cam_target_out(cv::Range(0,3), cv::Range(0,3)), CV_64F);
        tvec.convertTo(T_cam_target_out(cv::Range(0,3), cv::Range(3,4)), CV_64F);

        std::cout << "[PoseEstimator] ✓ solvePnP succeeded" << std::endl;
        std::cout << "[PoseEstimator] Translation: [" << tvec.at<double>(0) << ", " 
                  << tvec.at<double>(1) << ", " << tvec.at<double>(2) << "]" << std::endl;
        
        return true;
    }

    static cv::Mat quaternionToRotationMatrix(const cv::Vec4f &q) {
        float norm = std::sqrt(q[0]*q[0] + q[1]*q[1] + q[2]*q[2] + q[3]*q[3]);
        if (norm < 1e-8) {
            return cv::Mat::eye(3, 3, CV_64F);
        }

        float w = q[0]/norm, x = q[1]/norm, y = q[2]/norm, z = q[3]/norm;
        
        cv::Mat R = (cv::Mat_<double>(3,3) <<
            1 - 2*(y*y + z*z),  2*(x*y - z*w),    2*(x*z + y*w),
            2*(x*y + z*w),      1 - 2*(x*x + z*z),2*(y*z - x*w),
            2*(x*z - y*w),      2*(y*z + x*w),    1 - 2*(x*x + y*y)
        );
        return R;
    }

    bool computeTrackToImuWorld(const cv::Mat &T_cam_target, cv::Mat &T_track2imu_world_out) const {
        if (T_cam_target.empty()) {
            std::cerr << "[PoseEstimator] ✗ T_cam_target is empty" << std::endl;
            return false;
        }

        std::cout << "[PoseEstimator] Computing transform chain..." << std::endl;

        // Camera to Gimbal
        cv::Mat T_cam2gimbal = cv::Mat::eye(4,4,CV_64F);
        if (!config_.R_camera2gimbal.empty())
            config_.R_camera2gimbal.convertTo(T_cam2gimbal(cv::Range(0,3), cv::Range(0,3)), CV_64F);
        if (!config_.t_camera2gimbal.empty())
            config_.t_camera2gimbal.convertTo(T_cam2gimbal(cv::Range(0,3), cv::Range(3,4)), CV_64F);

        // Gimbal to IMU
        cv::Mat T_gimbal2imu = cv::Mat::eye(4,4,CV_64F);
        if (!config_.R_gimbal2imubody.empty())
            config_.R_gimbal2imubody.convertTo(T_gimbal2imu(cv::Range(0,3), cv::Range(0,3)), CV_64F);

        // IMU to World
        cv::Mat R_imu2world = quaternionToRotationMatrix(config_.imu_q);
        cv::Mat T_imu2world = cv::Mat::eye(4,4,CV_64F);
        R_imu2world.copyTo(T_imu2world(cv::Range(0,3), cv::Range(0,3)));

        // 完整变换链: World = T_imu2world * T_gimbal2imu * T_cam2gimbal * T_cam_target
        T_track2imu_world_out = T_imu2world * T_gimbal2imu * T_cam2gimbal * T_cam_target;
        
        std::cout << "[PoseEstimator] ✓ Transform chain computed" << std::endl;
        return true;
    }

    bool process(const cv::Mat &image,
                 cv::Mat &T_cam_target_out, 
                 cv::Mat &T_track2imu_world_out,
                 bool visualize = true) const
    {
        std::cout << "\n[PoseEstimator] Starting pose estimation..." << std::endl;
        std::cout << "========================================" << std::endl;

        // ✅ 检查输入
        if (config_.image_corners.size() != 4) {
            std::cerr << "[PoseEstimator] ✗ Need exactly 4 image corners, got " 
                      << config_.image_corners.size() << std::endl;
            return false;
        }

        // ✅ 计算3D角点
        ArmorCorners3D corners3d = computeArmorCorners3D();
        std::vector<cv::Point3f> object_pts = {
            corners3d.lt, corners3d.rt, corners3d.rb, corners3d.lb
        };
        
        std::cout << "\n[3D Object Points (Armor in world frame)]" << std::endl;
        std::cout << "  Left-Top:     " << corners3d.lt << std::endl;
        std::cout << "  Right-Top:    " << corners3d.rt << std::endl;
        std::cout << "  Right-Bottom: " << corners3d.rb << std::endl;
        std::cout << "  Left-Bottom:  " << corners3d.lb << std::endl;

        std::cout << "\n[2D Image Points (pixels)]" << std::endl;
        for (size_t i = 0; i < config_.image_corners.size(); ++i) {
            const char* names[] = {"Left-Top", "Right-Top", "Right-Bottom", "Left-Bottom"};
            std::cout << "  " << names[i] << ":     " << config_.image_corners[i] << std::endl;
        }

        // ✅ 可视化
        if (visualize && !image.empty()) {
            cv::Mat display = image.clone();
            
            // 绘制角点
            for (size_t i = 0; i < config_.image_corners.size(); ++i) {
                cv::circle(display, config_.image_corners[i], 8, cv::Scalar(0, 0, 255), -1);
                cv::circle(display, config_.image_corners[i], 10, cv::Scalar(0, 255, 255), 2);
                
                std::string label = std::to_string(i);
                cv::putText(display, label, 
                           config_.image_corners[i] + cv::Point2f(15, -15),
                           cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(255, 255, 255), 2);
            }
            
            // 连接角点形成装甲板
            for (size_t i = 0; i < 4; ++i) {
                cv::line(display, config_.image_corners[i], 
                        config_.image_corners[(i+1)%4], 
                        cv::Scalar(0, 255, 0), 3);
            }
            
            // 标注装甲板中心
            cv::Point2f center = (config_.image_corners[0] + config_.image_corners[2]) * 0.5f;
            cv::putText(display, "ARMOR", center,
                       cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 255, 255), 2);
            
            cv::imwrite("visualization.jpg", display);
            std::cout << "\n[PoseEstimator] ✓ Saved visualization.jpg" << std::endl;
        }

        // ✅ 求解 PnP
        std::cout << "\n[PnP Solving]" << std::endl;
        if (!solvePnP(object_pts, config_.image_corners, T_cam_target_out)) {
            return false;
        }

        // ✅ 计算完整变换
        std::cout << "\n[Transform Chain]" << std::endl;
        if (!computeTrackToImuWorld(T_cam_target_out, T_track2imu_world_out)) {
            return false;
        }

        std::cout << "\n========================================" << std::endl;
        std::cout << "[PoseEstimator] ✓ Pose estimation completed!" << std::endl;
        return true;
    }

private:
    Config config_;
};

} // namespace armor_pose

// --------------------------- Main ---------------------------
int main(int argc, char** argv) {
    std::cout << "========================================" << std::endl;
    std::cout << "Armor Pose Estimation (Direct Input)" << std::endl;
    std::cout << "========================================" << std::endl;

    std::string cfg_path = "../config.yaml";
    std::string image_path = "../test_image.jpg";

    if (argc >= 2) cfg_path = argv[1];
    if (argc >= 3) image_path = argv[2];

    std::cout << "[MAIN] Config: " << cfg_path << std::endl;
    std::cout << "[MAIN] Image:  " << image_path << std::endl;

    // ===== 加载配置 =====
    armor_pose::Config cfg;
    if (!armor_pose::ConfigLoader::loadFromYaml(cfg_path, cfg)) {
        std::cerr << "[MAIN] ✗ Failed to load config" << std::endl;
        return -1;
    }

    // ===== 读取图像（可选，仅用于可视化） =====
    cv::Mat image = cv::imread(image_path);
    if (image.empty()) {
        std::cout << "[MAIN] ⚠ Image not found, continuing without visualization" << std::endl;
    } else {
        std::cout << "[MAIN] ✓ Image loaded: " << image.cols << "x" << image.rows << std::endl;
    }

    // ===== 创建估计器并处理 =====
    armor_pose::PoseEstimator estimator(cfg);

    cv::Mat T_cam_target, T_track2imu_world;
    if (estimator.process(image, T_cam_target, T_track2imu_world, true)) {
        std::cout << "\n========================================" << std::endl;
        std::cout << "           RESULTS" << std::endl;
        std::cout << "========================================" << std::endl;
        
        std::cout << "\n[T_cam_target] Target pose in Camera frame:" << std::endl;
        std::cout << T_cam_target << std::endl;
        
        std::cout << "\n[T_track2imu_world] Target pose in World frame:" << std::endl;
        std::cout << T_track2imu_world << std::endl;
        
        // 提取位置和旋转
        cv::Mat tvec = T_track2imu_world(cv::Range(0,3), cv::Range(3,4));
        cv::Mat R = T_track2imu_world(cv::Range(0,3), cv::Range(0,3));
        
        std::cout << "\n[World Position] (x, y, z):" << std::endl;
        std::cout << "  x = " << tvec.at<double>(0) << " m" << std::endl;
        std::cout << "  y = " << tvec.at<double>(1) << " m" << std::endl;
        std::cout << "  z = " << tvec.at<double>(2) << " m" << std::endl;
        
        // 转换为欧拉角（ZYX顺序）
        double sy = std::sqrt(R.at<double>(0,0) * R.at<double>(0,0) + 
                             R.at<double>(1,0) * R.at<double>(1,0));
        bool singular = sy < 1e-6;
        
        double roll, pitch, yaw;
        if (!singular) {
            roll  = std::atan2(R.at<double>(2,1), R.at<double>(2,2));
            pitch = std::atan2(-R.at<double>(2,0), sy);
            yaw   = std::atan2(R.at<double>(1,0), R.at<double>(0,0));
        } else {
            roll  = std::atan2(-R.at<double>(1,2), R.at<double>(1,1));
            pitch = std::atan2(-R.at<double>(2,0), sy);
            yaw   = 0;
        }
        
        std::cout << "\n[World Orientation] (roll, pitch, yaw):" << std::endl;
        std::cout << "  roll  = " << roll * 180.0 / M_PI << "°" << std::endl;
        std::cout << "  pitch = " << pitch * 180.0 / M_PI << "°" << std::endl;
        std::cout << "  yaw   = " << yaw * 180.0 / M_PI << "°" << std::endl;
        
        std::cout << "\n========================================" << std::endl;
        std::cout << "[MAIN] ✓ SUCCESS!" << std::endl;
        std::cout << "========================================" << std::endl;
        
    } else {
        std::cout << "\n========================================" << std::endl;
        std::cout << "[MAIN] ✗ FAILED" << std::endl;
        std::cout << "========================================" << std::endl;
        return -1;
    }

    return 0;
}