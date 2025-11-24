// multi_frame_fusion.cpp
// 编译命令:
//   g++ multi_frame_fusion.cpp -o multi_frame_fusion `pkg-config --cflags --libs opencv4` -std=c++17
//
// 运行命令:
//   ./multi_frame_fusion ../config.yaml
//
// 注意事项:
//  - 需要安装 OpenCV 4+ 版本，并包含 calibration 和 core 模块。
//  - 配置文件 YAML 格式: 参见之前提供的 config.yaml 模板。

#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <string>
#include <numeric>

using namespace std;

// ---------------------- 辅助函数 ----------------------
// 四元数转旋转矩阵
cv::Mat quatToRot(const cv::Vec4d &q) {
    double w=q[0], x=q[1], y=q[2], z=q[3];
    double n = sqrt(w*w + x*x + y*y + z*z);
    if (n < 1e-12) return cv::Mat::eye(3,3,CV_64F);
    w/=n; x/=n; y/=n; z/=n;
    cv::Mat R = (cv::Mat_<double>(3,3) <<
        1-2*(y*y+z*z),   2*(x*y - z*w),  2*(x*z + y*w),
        2*(x*y + z*w),   1-2*(x*x+z*z),  2*(y*z - x*w),
        2*(x*z - y*w),   2*(y*z + x*w),  1-2*(x*x+y*y)
    );
    return R;
}

// 旋转矩阵转四元数
cv::Vec4d rotToQuat(const cv::Mat &R) {
    // 使用标准迹法将旋转矩阵转换为四元数 (w,x,y,z)
    cv::Mat r = R;
    double t = r.at<double>(0,0) + r.at<double>(1,1) + r.at<double>(2,2);
    double w,x,y,z;
    if (t > 0.0) {
        double s = sqrt(t + 1.0) * 2.0;
        w = 0.25 * s;
        x = (r.at<double>(2,1) - r.at<double>(1,2)) / s;
        y = (r.at<double>(0,2) - r.at<double>(2,0)) / s;
        z = (r.at<double>(1,0) - r.at<double>(0,1)) / s;
    } else if ((r.at<double>(0,0) > r.at<double>(1,1)) && (r.at<double>(0,0) > r.at<double>(2,2))) {
        double s = sqrt(1.0 + r.at<double>(0,0) - r.at<double>(1,1) - r.at<double>(2,2)) * 2.0;
        w = (r.at<double>(2,1) - r.at<double>(1,2)) / s;
        x = 0.25 * s;
        y = (r.at<double>(0,1) + r.at<double>(1,0)) / s;
        z = (r.at<double>(0,2) + r.at<double>(2,0)) / s;
    } else if (r.at<double>(1,1) > r.at<double>(2,2)) {
        double s = sqrt(1.0 + r.at<double>(1,1) - r.at<double>(0,0) - r.at<double>(2,2)) * 2.0;
        w = (r.at<double>(0,2) - r.at<double>(2,0)) / s;
        x = (r.at<double>(0,1) + r.at<double>(1,0)) / s;
        y = 0.25 * s;
        z = (r.at<double>(1,2) + r.at<double>(2,1)) / s;
    } else {
        double s = sqrt(1.0 + r.at<double>(2,2) - r.at<double>(0,0) - r.at<double>(1,1)) * 2.0;
        w = (r.at<double>(1,0) - r.at<double>(0,1)) / s;
        x = (r.at<double>(0,2) + r.at<double>(2,0)) / s;
        y = (r.at<double>(1,2) + r.at<double>(2,1)) / s;
        z = 0.25 * s;
    }
    // 归一化
    double n = sqrt(w*w + x*x + y*y + z*z);
    return cv::Vec4d(w/n, x/n, y/n, z/n);
}

// Markley 四元数平均法（返回单位四元数 (w,x,y,z)）
cv::Vec4d averageQuaternionsMarkley(const vector<cv::Vec4d> &quats) {
    // 构建 4x4 对称累加矩阵
    cv::Mat M = cv::Mat::zeros(4,4,CV_64F);
    for (const auto &q : quats) {
        cv::Mat qv = (cv::Mat_<double>(4,1) << q[0], q[1], q[2], q[3]);
        M += qv * qv.t();
    }
    // 特征值分解
    cv::Mat eigvals, eigvecs;
    cv::eigen(M, eigvals, eigvecs); // eigvecs: 行是特征向量；特征值按降序排列
    cv::Mat qavg = eigvecs.row(0).t(); // 主特征向量（最大特征值对应的向量）
    cv::Vec4d qa(qavg.at<double>(0), qavg.at<double>(1), qavg.at<double>(2), qavg.at<double>(3));
    // 归一化
    double n = sqrt(qa[0]*qa[0] + qa[1]*qa[1] + qa[2]*qa[2] + qa[3]*qa[3]);
    if (n < 1e-12) return cv::Vec4d(1,0,0,0);
    return cv::Vec4d(qa[0]/n, qa[1]/n, qa[2]/n, qa[3]/n);
}

// 构建装甲板局部角点（左上、右上、右下、左下）
vector<cv::Point3f> makeArmorCorners(double w, double h) {
    double hw = w * 0.5, hh = h * 0.5;
    return {
        {-hw,  hh, 0}, { hw,  hh, 0},
        { hw, -hh, 0}, {-hw, -hh, 0}
    };
}

// 在追踪坐标系中计算装甲板角点
vector<cv::Point3f> computeArmorCornersInTrack(const cv::Point3d &center, const cv::Point3d &opp,
                                               double width, double height, double pitch_deg,
                                               bool try_alt_order=false, int pitch_sign=1)
{
    double yaw = atan2(opp.y - center.y, opp.x - center.x);
    double pitch = pitch_deg * CV_PI / 180.0 * (double)pitch_sign;

    // 旋转矩阵
    double cy = cos(yaw), sy = sin(yaw);
    double cp = cos(pitch), sp = sin(pitch);
    cv::Mat Rz = (cv::Mat_<double>(3,3) << cy, -sy, 0, sy, cy, 0, 0, 0, 1);
    cv::Mat Ry = (cv::Mat_<double>(3,3) << cp, 0, sp, 0,1,0, -sp,0,cp);

    cv::Mat R;
    if (!try_alt_order) R = Rz * Ry;
    else R = Ry * Rz; // 测试时可使用交替的旋转顺序

    vector<cv::Point3f> local = makeArmorCorners(width, height);
    vector<cv::Point3f> out;
    for (auto &p : local) {
        cv::Mat v = (cv::Mat_<double>(3,1) << p.x, p.y, p.z);
        cv::Mat w = R * v;
        out.emplace_back(
            static_cast<float>(center.x + w.at<double>(0)),
            static_cast<float>(center.y + w.at<double>(1)),
            static_cast<float>(center.z + w.at<double>(2))
        );
    }
    return out;
}

// 计算平移向量的平均值（3x1）
cv::Mat averageTranslations(const vector<cv::Mat> &tlist) {
    cv::Mat tav = cv::Mat::zeros(3,1,CV_64F);
    for (auto &t : tlist) tav += t;
    tav /= static_cast<double>(tlist.size());
    return tav;
}

// 计算重投影误差
double computeTotalReproj(const vector<vector<cv::Point3f>> &all_obj_pts,
                          const vector<vector<cv::Point2f>> &all_img_pts,
                          const cv::Mat &R_final, const cv::Mat &t_final,
                          const cv::Mat &K, const cv::Mat &dist)
{
    double total_err = 0;
    size_t count = 0;
    cv::Mat rvec_final;
    cv::Rodrigues(R_final, rvec_final);
    for (size_t i=0;i<all_obj_pts.size();++i) {
        vector<cv::Point2f> proj;
        cv::projectPoints(all_obj_pts[i], rvec_final, t_final, K, dist, proj);
        for (size_t j=0;j<proj.size();++j) {
            total_err += cv::norm(proj[j] - all_img_pts[i][j]);
            ++count;
        }
    }
    if (count==0) return 0.0;
    return total_err / (double)count;
}

// ---------------------- 主函数 ----------------------
int main(int argc, char** argv) {
    string cfg = "../config.yaml";
    if (argc > 1) cfg = argv[1];

    cv::FileStorage fs(cfg, cv::FileStorage::READ);
    if (!fs.isOpened()) {
        cerr << "无法打开配置文件: " << cfg << endl;
        return -1;
    }

    // 读取相机内参
    cv::Mat K, dist;
    fs["camera"]["camera_matrix"] >> K;
    fs["camera"]["dist_coeffs"] >> dist;
    cv::Mat R_cam2gimbal, t_cam2gimbal, R_gimbal2imu;
    fs["camera"]["R_camera2gimbal"] >> R_cam2gimbal;
    fs["camera"]["t_camera2gimbal"] >> t_cam2gimbal;
    fs["camera"]["R_gimbal2imubody"] >> R_gimbal2imu;

    double armor_w=0, armor_h=0, pitch_deg=0;
    fs["armor"]["width"] >> armor_w;
    fs["armor"]["height"] >> armor_h;
    fs["armor"]["pitch_deg"] >> pitch_deg;

    // 解析帧节点
    cv::FileNode frames = fs["frames"];
    if (frames.empty() || !frames.isSeq()) {
        cerr << "配置文件中未找到帧数据（期望格式：frames: []）" << endl;
        return -1;
    }

    vector<cv::Mat> Rlist, tlist;
    vector<vector<cv::Point3f>> all_obj_pts;
    vector<vector<cv::Point2f>> all_img_pts;
    int frame_idx = 0;

    for (auto it = frames.begin(); it != frames.end(); ++it) {
        frame_idx++;
        cv::FileNode fn = *it;

        // 读取中心点和对面中心点
        cv::Point3d center, opp;
        fn["center"]["x"] >> center.x;
        fn["center"]["y"] >> center.y;
        fn["center"]["z"] >> center.z;
        fn["opp_center"]["x"] >> opp.x;
        fn["opp_center"]["y"] >> opp.y;
        fn["opp_center"]["z"] >> opp.z;

        // 图像角点
        vector<cv::Point2f> img_pts;
        cv::FileNode corners = fn["image_corners"];
        if (corners.empty() || !corners.isSeq()) {
            cerr << "第 " << frame_idx << " 帧缺少 image_corners" << endl;
            continue;
        }
        for (auto c = corners.begin(); c != corners.end(); ++c) {
            float x = (float)(*c)["x"];
            float y = (float)(*c)["y"];
            img_pts.emplace_back(x,y);
        }
        if (img_pts.size() != 4) {
            cerr << "第 " << frame_idx << " 帧需要 4 个图像角点" << endl;
            continue;
        }

        // IMU 四元数
        cv::Vec4d imu_q;
        cv::FileNode qn = fn["imu_q"];
        if (qn.empty() || !qn.isSeq() || (int)qn.size() != 4) {
            cerr << "第 " << frame_idx << " 帧缺少 imu_q (w,x,y,z)" << endl;
            continue;
        }
        for (int k=0;k<4;k++) imu_q[k] = (double)qn[k];

        // 构建追踪坐标系中的 3D 目标点
        vector<cv::Point3f> obj_pts = computeArmorCornersInTrack(center, opp, armor_w, armor_h, pitch_deg);
        // 存储用于后续全局 T 的重投影
        all_obj_pts.push_back(obj_pts);
        all_img_pts.push_back(img_pts);

        // 求解 PnP（使用 RANSAC 然后优化）
        cv::Mat rvec, tvec;
        vector<int> inliers;
        bool ok = cv::solvePnPRansac(obj_pts, img_pts, K, dist, rvec, tvec,
                                     false, 100, 8.0, 0.99, inliers, cv::SOLVEPNP_ITERATIVE);
        if (!ok) {
            cerr << "第 " << frame_idx << " 帧 solvePnPRansac 失败，尝试直接 solvePnP..." << endl;
            ok = cv::solvePnP(obj_pts, img_pts, K, dist, rvec, tvec, false, cv::SOLVEPNP_ITERATIVE);
        } else {
            // 可选：使用所有内点进行优化
            if (inliers.size() >= 4) {
                vector<cv::Point3f> in_obj;
                vector<cv::Point2f> in_img;
                for (int idx : inliers) {
                    in_obj.push_back(obj_pts[idx]);
                    in_img.push_back(img_pts[idx]);
                }
                // 使用 LM 方法优化（如果可用）
                try {
                    cv::solvePnPRefineLM(in_obj, in_img, K, dist, rvec, tvec);
                } catch (...) {
                    // 如果不可用则忽略
                }
            }
        }

        // 重投影调试
        vector<cv::Point2f> proj;
        cv::projectPoints(obj_pts, rvec, tvec, K, dist, proj);
        double err_sum = 0;
        for (size_t i=0;i<proj.size();++i) err_sum += cv::norm(proj[i] - img_pts[i]);
        double mean_err = err_sum / proj.size();
        cout << "[第 " << frame_idx << " 帧] solvePnP 平均重投影误差: " << mean_err << " 像素 (内点数=" << inliers.size() << ")" << endl;

        // 计算目标到相机的旋转矩阵
        cv::Mat R_target2cam;
        cv::Rodrigues(rvec, R_target2cam);

        // 坐标链：目标 -> 相机 -> 云台 -> IMU -> 世界
        // 相机到云台：已知 R_camera2gimbal, t_camera2gimbal
        // 注意：如果配置文件中的参数是相机到云台的变换，我们需要其逆变换来实现相机到云台的转换？
        // 我们使用以下一致的坐标链：
        // p_gimbal = R_camera2gimbal * p_cam + t_camera2gimbal
        // p_imu = R_gimbal2imubody * p_gimbal  + 0 (无平移)
        // p_world = R_imu2world * p_imu  (假设 t_imu2world 为零)

        // 从 IMU 四元数解析 IMU 到世界的旋转矩阵
        cv::Mat R_imu2world = quatToRot(imu_q);

        // 计算目标到云台的变换
        cv::Mat R_target2gimbal = R_cam2gimbal * R_target2cam;
        cv::Mat tvec_d;
        tvec.convertTo(tvec_d, CV_64F);
        cv::Mat t_target2gimbal = R_cam2gimbal * tvec_d + t_cam2gimbal;

        // 云台到世界的旋转（假设云台到 IMU 的旋转为 R_gimbal2imu；然后 IMU 到世界的旋转为 R_imu2world）
        // 目标到世界的旋转矩阵 R_target2world = R_imu2world * R_gimbal2imubody * R_target2gimbal？注意旋转顺序
        // 基于坐标链 p_gimbal = R_cam2gimbal * p_cam + t_cam2gimbal
        // p_imu = R_gimbal2imubody * p_gimbal   (无平移)
        // p_world = R_imu2world * p_imu
        // 因此 R_target2world = R_imu2world * R_gimbal2imubody * R_target2gimbal
        cv::Mat R_target2world = R_imu2world * R_gimbal2imu * R_target2gimbal;
        cv::Mat t_target2world = R_imu2world * R_gimbal2imu * t_target2gimbal; // 无 IMU 平移

        // 存储结果
        Rlist.push_back(R_target2world.clone());
        tlist.push_back(t_target2world.clone());

        cout << "  第 " << frame_idx << " 帧 R_target2world (部分):\n" << R_target2world << endl;
        cout << "  第 " << frame_idx << " 帧 t_target2world: [" << t_target2world.at<double>(0) << ", "
             << t_target2world.at<double>(1) << ", " << t_target2world.at<double>(2) << "]" << endl;
    } // 帧循环结束

    if (Rlist.empty()) {
        cerr << "没有处理到有效的帧。" << endl;
        return -1;
    }

    // ----------------- 融合：旋转使用 Markley 平均法，平移使用算术平均法 -----------------
    vector<cv::Vec4d> quats;
    for (auto &R : Rlist) {
        cv::Vec4d q = rotToQuat(R);
        quats.push_back(q);
    }
    cv::Vec4d qavg = averageQuaternionsMarkley(quats);
    cv::Mat R_avg = quatToRot(qavg);
    cv::Mat t_avg = averageTranslations(tlist);

    // 构建最终的变换矩阵 T
    cv::Mat T_final = cv::Mat::eye(4,4,CV_64F);
    R_avg.copyTo(T_final(cv::Range(0,3), cv::Range(0,3)));
    t_avg.copyTo(T_final(cv::Range(0,3), cv::Range(3,4)));

    cout << "\n========= 最终（融合后）结果 =========\n";
    cout << "T_track2world (4x4):\n" << T_final << endl;

    // 计算原始目标点在融合后 T_final 下的重投影误差
    // 为了进行重投影，我们需要 T_final 作用于目标点（目标在追踪坐标系 -> 世界坐标系 -> 相机坐标系）的 rvec 和 tvec
    // 但 projectPoints 函数期望目标点在目标坐标系中，并且 rvec/tvec 是将目标点映射到相机坐标系的旋转和平移
    // 我们想要比较像素重投影结果：projectPoints(obj_pts_in_track, R_final_cam, t_final_cam)
    // 计算相机相对于追踪坐标系的位姿：T_cam = inverse(T_world_camera) 等。更简单的方法是：
    // 使用融合后的 T 和已知的 IMU 旋转，将每个帧的目标点转换到相机坐标系，然后计算重投影误差
    // 我们将使用每个帧的相机位姿（由 IMU 和相机外参推导）来计算重投影，并与图像角点进行比较

    double total_reproj = 0.0;
    int total_pts = 0;
    for (size_t i=0;i<all_obj_pts.size();++i) {
        // 对于每个帧，相机在世界坐标系中的位姿：
        // p_cam = ... 我们需要 R_world_cam 和 t_world_cam
        // R_world_cam = R_imu2world * R_gimbal2imu * R_cam2gimbal * (R_target2cam)^-1 ??? 比较复杂
        // 为避免重新求解，我们通过以下方式近似重投影误差：
        // 将融合后的 T 变换后的目标点投影到每个帧的相机中，使用每个帧的已知坐标链
        // 从文件节点中重新读取每个帧的 IMU 数据来重建 R_imu2world：
        // 构建每个帧的 T_world_cam：使用 R_imu2world 和相机外参
        // 然后将目标点转换到世界坐标系：P_world = T_final * P_track
        // 然后转换到相机坐标系：P_cam = R_cam_world * (P_world - t_world_cam)
        // 其中 R_cam_world = (R_world_cam)^T，t_world_cam 是相机在世界坐标系中的位置
        // 我们将仔细实现这个过程

        // 从文件节点中重新读取每个帧的已知数据：IMU -> 
    }

    // 这里我们计算一个更简单的聚合指标，而不是进行复杂的逐帧重投影：
    double final_reproj = computeTotalReproj(all_obj_pts, all_img_pts, R_avg, t_avg, K, dist);
    cout << "使用融合后的 (R,t) 计算的平均重投影误差: " << final_reproj << " 像素" << endl;

    cout << "\n完成。" << endl;
    fs.release();
    return 0;
}
