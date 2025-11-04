#pragma once

#include <rclcpp/rclcpp.hpp>
#include <visualization_msgs/msg/marker.hpp>
#include <visualization_msgs/msg/marker_array.hpp>
#include <geometry_msgs/msg/point_stamped.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <geometry_msgs/msg/pose_with_covariance_stamped.hpp>
#include <sensor_msgs/msg/joint_state.hpp>
#include <std_msgs/msg/string.hpp>
#include <tf2_ros/transform_broadcaster.h>
#include <tf2_ros/static_transform_broadcaster.h>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>
#include <Eigen/Dense>
#include <memory>
#include <string>
#include <vector>

namespace io
{

class VisualizationPublisher : public rclcpp::Node
{
public:
    VisualizationPublisher();
    ~VisualizationPublisher();

    // 目标相关可视化
    void publish_target_marker(const Eigen::Vector3d& position, const std::string& target_id = "target");
    void publish_target_trajectory(const std::vector<Eigen::Vector3d>& trajectory);
    void publish_target_velocity_arrow(const Eigen::Vector3d& position, const Eigen::Vector3d& velocity);
    void publish_target_uncertainty_ellipsoid(const Eigen::Vector3d& position, const Eigen::Matrix3d& covariance);

    // 云台相关可视化
    void publish_gimbal_pose(const Eigen::Vector3d& position, const Eigen::Quaterniond& orientation);
    void publish_gimbal_fov(const Eigen::Vector3d& position, const Eigen::Quaterniond& orientation, 
                           double fov_angle = 60.0, double range = 10.0);
    void publish_gimbal_trajectory(const std::vector<Eigen::Vector3d>& trajectory);

    // 机器人状态可视化
    void publish_robot_state(const Eigen::Vector3d& position, const Eigen::Quaterniond& orientation,
                            double yaw, double pitch);
    void publish_robot_urdf(const std::string& urdf_content);
    void publish_joint_states(const std::vector<std::string>& joint_names,
                             const std::vector<double>& joint_positions);

    // 环境可视化
    void publish_ground_plane(double size = 20.0);
    void publish_coordinate_frames();
    void publish_camera_frustum(const Eigen::Vector3d& position, const Eigen::Quaterniond& orientation,
                               double fov_x, double fov_y, double range);

    // 路径和轨迹可视化
    void publish_path(const std::vector<Eigen::Vector3d>& path, const std::string& frame_id = "world");
    void publish_trajectory_prediction(const std::vector<Eigen::Vector3d>& predicted_points);

    // 清除标记
    void clear_all_markers();
    void clear_marker_by_id(int marker_id);

    // 设置参数
    void set_frame_id(const std::string& frame_id) { frame_id_ = frame_id; }
    void set_marker_lifetime(double lifetime) { marker_lifetime_ = rclcpp::Duration::from_seconds(lifetime); }

private:
    // 发布器
    rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr marker_pub_;
    rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr marker_array_pub_;
    rclcpp::Publisher<geometry_msgs::msg::PointStamped>::SharedPtr point_pub_;
    rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr pose_pub_;
    rclcpp::Publisher<sensor_msgs::msg::JointState>::SharedPtr joint_state_pub_;
    rclcpp::Publisher<std_msgs::msg::String>::SharedPtr robot_description_pub_;

    // TF广播器
    std::unique_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_;
    std::unique_ptr<tf2_ros::StaticTransformBroadcaster> static_tf_broadcaster_;

    // 参数
    std::string frame_id_;
    rclcpp::Duration marker_lifetime_;
    int marker_id_counter_;

    // 辅助方法
    visualization_msgs::msg::Marker create_marker(const std::string& ns, int id, int type);
    void publish_tf_transform(const std::string& parent_frame, const std::string& child_frame,
                            const Eigen::Vector3d& translation, const Eigen::Quaterniond& rotation);
};

} // namespace io
