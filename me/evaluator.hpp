#ifndef ME_EVALUATOR_HPP
#define ME_EVALUATOR_HPP

#include <vector>
#include <cstdint>
#include <Eigen/Dense>
#include <string>
#include <yaml-cpp/yaml.h>

namespace me {

// 与发送端一致：帧头/帧尾均为4字节
#pragma pack(push, 1)
struct BluetoothFrame {
    int32_t frame_head;            // 帧头：0x000000FF
    float   coordinates_data[4][3];// 4个点(x,y,z)，共48字节
    int32_t frame_tail;            // 帧尾：0x000000EF
};
#pragma pack(pop)

class Evaluator {
public:
    Evaluator();
    //explicit Evaluator(const std::string& config_path);
    ~Evaluator();

    // 初始化蓝牙串口（需在 runOnce 前调用）
    bool initBluetooth(const std::string& port = "/dev/rfcomm0", int baudrate = 921600);

    // 一次完整流程：蓝牙接收 -> 坐标变换 -> 温漂补偿 -> 误差评估
    // predicted 为小电脑识别/预测的坐标（温漂后小云台坐标系）
    // errors_out 输出对应点误差（predicted - ground_truth_after_drift）
    bool runOnce(const std::vector<Eigen::Vector3d>& predicted,
                 std::vector<Eigen::Vector3d>& errors_out);

    // 1) 蓝牙数据接收模块：输出靶车坐标（靶车坐标系）
    bool recvBluetooth(std::vector<Eigen::Vector3d>& in_target_cs);

    // 2) 坐标变换模块：靶车系 -> 温漂前小云台坐标系
    bool transform(const std::vector<Eigen::Vector3d>& in_target_cs,
                   std::vector<Eigen::Vector3d>& out_gimbal_pre_drift);

    // 3) 温漂补偿模块：温漂前 -> 温漂后小云台坐标系
    bool applyTem(const std::vector<Eigen::Vector3d>& in_gimbal_pre_drift,
                  std::vector<Eigen::Vector3d>& out_gimbal_after_drift);

    // 4) 误差评估模块：predicted(温漂后) 与 ground-truth(温漂后) 的误差
    bool evaluateError(const std::vector<Eigen::Vector3d>& gt_gimbal_after_drift,
                       const std::vector<Eigen::Vector3d>& pred_gimbal_after_drift,
                       std::vector<Eigen::Vector3d>& errors_out);

    // 辅助函数：从缓冲区中查找并解析完整帧
    bool findAndParseFrame(BluetoothFrame& frame_out);

private:
    // 坐标变换矩阵：靶车系 -> 温漂前小云台系
    Eigen::Matrix4d T_gimbal_pre__target_ = Eigen::Matrix4d::Identity();
    
    // 温漂拟合参数
    std::vector<double> drift_params_;
    
    // ✅ 蓝牙串口相关
    int bt_fd_ = -1;                          // 蓝牙文件描述符
    static constexpr int FRAME_SIZE = 56;     // 帧大小 (4+48+4=56字节)
    static constexpr int BUFFER_SIZE = 512;   // 接收缓冲区大小
    unsigned char buffer_[BUFFER_SIZE];       // 接收缓冲区
    int buffer_len_ = 0;                      // 缓冲区当前数据长度
};

} // namespace me

#endif // ME_EVALUATOR_HPP