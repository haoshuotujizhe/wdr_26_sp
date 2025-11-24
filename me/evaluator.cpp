#include "evaluator.hpp"
#include <iostream>
#include <fcntl.h>
#include <unistd.h>
#include <termios.h>
#include <cstring>

namespace me {
Evaluator::Evaluator(/*const std::string & config_path*/){
    // auto yaml = YAML::LoadFile(config_path);

    
}
Evaluator::~Evaluator() {
    if (bt_fd_ >= 0) {
        close(bt_fd_);
        bt_fd_ = -1;
        std::cout << "[Evaluator] Bluetooth connection closed" << std::endl;
    }
}


// 调用流程：按顺序串起四个大模块
bool Evaluator::runOnce(const std::vector<Eigen::Vector3d>& predicted,
                        std::vector<Eigen::Vector3d>& errors_out) {
    std::vector<Eigen::Vector3d> target_in_target_cs;
    if (!recvBluetooth(target_in_target_cs)) return false;

    std::vector<Eigen::Vector3d> target_in_gimbal_pre;
    if (!transform(target_in_target_cs, target_in_gimbal_pre)) return false;

    std::vector<Eigen::Vector3d> gt_in_gimbal_after;
    if (!applyTem(target_in_gimbal_pre, gt_in_gimbal_after)) return false;

    if (!evaluateError(gt_in_gimbal_after, predicted, errors_out)) return false;

    return true;
}


// ========== 蓝牙初始化 ==========
bool Evaluator::initBluetooth(const std::string& port, int baudrate) {
    bt_fd_ = open(port.c_str(), O_RDWR | O_NOCTTY);
    if (bt_fd_ == -1) {
        std::cerr << "[Evaluator] Failed to open " << port << std::endl;
        return false;
    }

    struct termios options;
    memset(&options, 0, sizeof(options));

    // 设置波特率
    speed_t baud;
    switch (baudrate) {
        case 9600:   baud = B9600;   break;
        case 115200: baud = B115200; break;
        case 921600: baud = B921600; break;
        default:
            std::cerr << "[Evaluator] Unsupported baudrate: " << baudrate << std::endl;
            close(bt_fd_);
            bt_fd_ = -1;
            return false;
    }
    
    cfsetispeed(&options, baud);
    cfsetospeed(&options, baud);

    // 设置串口参数：8N1，无流控
    options.c_cflag |= (CLOCAL | CREAD);
    options.c_cflag &= ~PARENB;   // 无校验位
    options.c_cflag &= ~CSTOPB;   // 1个停止位
    options.c_cflag &= ~CSIZE;
    options.c_cflag |= CS8;       // 8位数据位
    
    // 原始二进制模式
    options.c_lflag &= ~(ICANON | ECHO | ECHOE | ISIG);
    options.c_iflag &= ~(ISTRIP | INLCR | IGNCR | ICRNL | IXON | IXOFF | IXANY);
    options.c_oflag &= ~OPOST;

    options.c_cc[VMIN] = 0;   // 非阻塞
    options.c_cc[VTIME] = 10; // 1秒超时

    if (tcsetattr(bt_fd_, TCSANOW, &options) != 0) {
        std::cerr << "[Evaluator] Error setting serial port attributes" << std::endl;
        close(bt_fd_);
        bt_fd_ = -1;
        return false;
    }

    tcflush(bt_fd_, TCIOFLUSH);
    buffer_len_ = 0;

    std::cout << "[Evaluator] ✓ Bluetooth initialized on " << port 
              << " @ " << baudrate << " baud" << std::endl;
    return true;
}

// ========== 查找并解析完整帧 ==========
bool Evaluator::findAndParseFrame(BluetoothFrame& frame_out) {
    // 在缓冲区中查找完整帧
    if (buffer_len_ < FRAME_SIZE) {
        return false; // 数据不足一帧
    }

    int frame_start = -1;
    
    // 查找帧头 0x000000FF (小端序: FF 00 00 00)
    for (int i = 0; i <= buffer_len_ - FRAME_SIZE; i++) {
        if (buffer_[i]   == 0xFF && buffer_[i+1] == 0x00 && 
            buffer_[i+2] == 0x00 && buffer_[i+3] == 0x00) {
            
            // 检查对应位置的帧尾 0x000000EF (小端序: EF 00 00 00)
            int tail_pos = i + FRAME_SIZE - 4;
            if (tail_pos + 3 < buffer_len_ &&
                buffer_[tail_pos]   == 0xEF && buffer_[tail_pos+1] == 0x00 && 
                buffer_[tail_pos+2] == 0x00 && buffer_[tail_pos+3] == 0x00) {
                frame_start = i;
                break;
            }
        }
    }

    if (frame_start == -1) {
        // 未找到完整帧
        if (buffer_len_ > FRAME_SIZE + 10) {
            // 缓冲区有足够数据但没找到帧，移除第一个字节继续查找
            memmove(buffer_, buffer_ + 1, buffer_len_ - 1);
            buffer_len_--;
        }
        return false;
    }

    // 找到完整帧，复制数据
    memcpy(&frame_out, buffer_ + frame_start, FRAME_SIZE);

    // 验证帧头帧尾
    if (frame_out.frame_head != 0xFF || frame_out.frame_tail != 0xEF) {
        std::cerr << "[Evaluator] Frame validation failed: head=0x" 
                  << std::hex << frame_out.frame_head 
                  << ", tail=0x" << frame_out.frame_tail << std::dec << std::endl;
        
        // 移除第一个字节继续查找
        memmove(buffer_, buffer_ + 1, buffer_len_ - 1);
        buffer_len_--;
        return false;
    }

    // 移除已处理的帧
    int bytes_to_remove = frame_start + FRAME_SIZE;
    memmove(buffer_, buffer_ + bytes_to_remove, buffer_len_ - bytes_to_remove);
    buffer_len_ -= bytes_to_remove;

    return true;
}

// ========== 蓝牙数据接收 ==========
bool Evaluator::recvBluetooth(std::vector<Eigen::Vector3d>& in_target_cs) {
    if (bt_fd_ < 0) {
        std::cerr << "[Evaluator] Bluetooth not initialized" << std::endl;
        return false;
    }

    int max_attempts = 100;
    int attempts = 0;
    
    while (attempts++ < max_attempts) {
        // 尝试从缓冲区解析帧
        BluetoothFrame frame;
        if (findAndParseFrame(frame)) {
            std::cout << "[Evaluator] ✓ Frame received" << std::endl;
            
            in_target_cs.clear();
            in_target_cs.reserve(4);
            
            for (int i = 0; i < 4; i++) {
                Eigen::Vector3d pt(
                    static_cast<double>(frame.coordinates_data[i][0]),
                    static_cast<double>(frame.coordinates_data[i][1]),
                    static_cast<double>(frame.coordinates_data[i][2])
                );
                in_target_cs.push_back(pt);
                
                std::cout << "  Point[" << i << "]: (" 
                          << pt.x() << ", " << pt.y() << ", " << pt.z() << ")" << std::endl;
            }
            
            return true;
        }

        // 缓冲区数据不足，读取更多数据
        int n = read(bt_fd_, buffer_ + buffer_len_, BUFFER_SIZE - buffer_len_);
        
        if (n > 0) {
            buffer_len_ += n;
            std::cout << "[Evaluator] Received " << n << " bytes, buffer total: " 
                      << buffer_len_ << std::endl;
            // ✅ 收到数据后立即尝试解析，不要 continue
        } else if (n == 0) {
            // ✅ 删除：usleep(1000);
            // 串口超时会自动等待 VTIME (1秒)，不需要额外 sleep
            // 继续循环尝试读取
        } else {
            std::cerr << "[Evaluator] ✗ Read error: " << strerror(errno) << std::endl;
            return false;
        }

        // ✅ 修改溢出保护：只在真正没找到帧头时才清空
        if (buffer_len_ >= BUFFER_SIZE - 100) {
            // 尝试查找帧头
            bool found_header = false;
            for (int i = 0; i < buffer_len_ - 4; i++) {
                if (buffer_[i]   == 0xFF && buffer_[i+1] == 0x00 && 
                    buffer_[i+2] == 0x00 && buffer_[i+3] == 0x00) {
                    found_header = true;
                    // 保留从帧头开始的数据
                    memmove(buffer_, buffer_ + i, buffer_len_ - i);
                    buffer_len_ -= i;
                    std::cout << "[Evaluator] Found header, keeping " << buffer_len_ 
                              << " bytes" << std::endl;
                    break;
                }
            }
            
            if (!found_header) {
                std::cout << "[Evaluator] Buffer overflow, no header found, clearing" 
                          << std::endl;
                buffer_len_ = 0;
            }
        }
    }

    std::cerr << "[Evaluator] ✗ Failed to receive frame after " 
              << max_attempts << " attempts" << std::endl;
    return false;
}


bool Evaluator::transform(const std::vector<Eigen::Vector3d>& in_target_cs,
                          std::vector<Eigen::Vector3d>& out_gimbal_pre_drift) {
    if (in_target_cs.empty()) {
        std::cerr << "[Evaluator] transform: Input is empty" << std::endl;
        return false;
    }
    
    if (in_target_cs.size() != 4) {
        std::cerr << "[Evaluator] transform: Expected 4 points, got " 
                  << in_target_cs.size() << std::endl;
        return false;
    }

    // ✅ 定义变换矩阵 T_gimbal_pre__target_
    // 格式：4×4 齐次变换矩阵
    //   [R11, R12, R13, tx]
    //   [R21, R22, R23, ty]
    //   [R31, R32, R33, tz]
    //   [  0,   0,   0,  1]
    Eigen::Matrix4d T_gimbal_pre__target_;
    T_gimbal_pre__target_ << 
        -0.6391637787003157, -0.02718197399938643, -0.7685901406387093, 2.135206642458094,
        -0.7641640116092752, 0.1351554438565508, 0.6307030754302364, 0.6244126834731989,
        0.08673538700411587, 0.9904514860837131, -0.1071579505021016, 0.2517216272863683,
        0, 0, 0, 1;

    std::cout << "[Evaluator] Transform matrix T_gimbal_pre__target_:" << std::endl;
    std::cout << T_gimbal_pre__target_ << std::endl;

    // ✅ 将输入的 vector 转换为 4×3 矩阵 (行优先)
    // 每一行是一个点的 (x, y, z)
    Eigen::Matrix<double, 4, 3> in_target_cs_matrix;
    for (int i = 0; i < 4; i++) {
        in_target_cs_matrix.row(i) = in_target_cs[i].transpose();
    }
    
    std::cout << "[Evaluator] Input points in target coordinate system:" << std::endl;
    std::cout << in_target_cs_matrix << std::endl;

    // ✅ 齐次化：4×3 -> 4×4 (最后一列补1)
    // 格式：
    //   [x0, y0, z0, 1]
    //   [x1, y1, z1, 1]
    //   [x2, y2, z2, 1]
    //   [x3, y3, z3, 1]
    Eigen::Matrix4d in_target_cs_homo;
    in_target_cs_homo.leftCols<3>() = in_target_cs_matrix;  // 前3列是xyz
    in_target_cs_homo.col(3) = Eigen::Vector4d::Ones();     // 第4列全是1

    // std::cout << "[Evaluator] Homogeneous coordinates (before transform):" << std::endl;
    // std::cout << in_target_cs_homo << std::endl;

    // ✅ 应用变换：result = T * points^T
    // 注意：需要转置，因为变换矩阵左乘列向量
    // T (4×4) × points^T (4×4) = result^T (4×4)
    Eigen::Matrix4d out_gimbal_pre_homo = T_gimbal_pre__target_ * in_target_cs_homo.transpose();
    
    // 转置回来：4×4 -> 4×4 (每行是一个变换后的点)
    out_gimbal_pre_homo.transposeInPlace();

    //std::cout << "[Evaluator] Homogeneous coordinates (after transform):" << std::endl;
    //std::cout << out_gimbal_pre_homo << std::endl;

    // ✅ 去齐次化：4×4 -> 4×3 (取前3列)
    Eigen::Matrix<double, 4, 3> out_gimbal_pre_matrix = out_gimbal_pre_homo.leftCols<3>();

    std::cout << "[Evaluator] Output points in gimbal coordinate system:" << std::endl;
    std::cout << out_gimbal_pre_matrix << std::endl;

    // // ✅ 转换回 vector<Eigen::Vector3d> 格式
    // out_gimbal_pre_drift.clear();
    // out_gimbal_pre_drift.reserve(4);
    
    // for (int i = 0; i < 4; i++) {
    //     out_gimbal_pre_drift.push_back(out_gimbal_pre_matrix.row(i).transpose());
        
    //     std::cout << "  Transformed Point[" << i << "]: (" 
    //               << out_gimbal_pre_drift[i].x() << ", "
    //               << out_gimbal_pre_drift[i].y() << ", "
    //               << out_gimbal_pre_drift[i].z() << ")" << std::endl;
    // }

    std::cout << "======[Evaluator] ✓ Coordinate transformation completed======" << std::endl;
    return true;
}

bool Evaluator::applyTem(const std::vector<Eigen::Vector3d>& in_gimbal_pre_drift,
                                  std::vector<Eigen::Vector3d>& out_gimbal_after_drift) {
    // TODO: 根据 drift_params_ 和当前温度，对坐标叠加温漂影响
    return true;
}

bool Evaluator::evaluateError(const std::vector<Eigen::Vector3d>& gt_gimbal_after_drift,
                              const std::vector<Eigen::Vector3d>& pred_gimbal_after_drift,
                              std::vector<Eigen::Vector3d>& errors_out) {
    // TODO: errors_out[i] = pred[i] - gt[i]
    return true;
}

} // namespace me
int main(){
    me::Evaluator evaluator;
    
    if (!evaluator.initBluetooth("/dev/rfcomm0", 921600)) {
        std::cerr << "Failed to initialize bluetooth" << std::endl;
        return -1;
    }
    
    std::cout << "\n========== Starting continuous reception ==========" << std::endl;
    
    while (true) {
        std::vector<Eigen::Vector3d> in_target_cs, out_gimbal_pre_drift;
        
        std::cout << "\n---------- Waiting for new frame ----------" << std::endl;
        
        if (evaluator.recvBluetooth(in_target_cs)) {
            evaluator.transform(in_target_cs, out_gimbal_pre_drift);
            
            // TODO: 继续调用 applyTem 和 evaluateError
            // evaluator.applyTem(...);
            // evaluator.evaluateError(...);
            
            std::cout << "---------- Frame processed successfully ----------\n" << std::endl;
        } else {
            std::cerr << "Failed to receive frame, retrying..." << std::endl;
        }
        
        // 可选：按 Ctrl+C 退出
        // 或者添加退出条件，例如：
        // char c;
        // std::cout << "Press 'q' to quit, any other key to continue: ";
        // std::cin >> c;
        // if (c == 'q') break;
    }
    
    std::cout << "\n========== Program terminated ==========" << std::endl;
    return 0;
}