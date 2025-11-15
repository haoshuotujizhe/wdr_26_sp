#include <iostream>
#include <fcntl.h>
#include <unistd.h>
#include <termios.h>
#include <cstring>
#include <cstdio>

// 定义与发送端相同的帧结构
#pragma pack(push, 1)
typedef struct {
    int32_t frame_head;           // 帧头 0xFF，4字节
    float coordinates_data[4][3]; // 坐标数组，4×3×4=48字节
    int32_t frame_tail;           // 帧尾 0xEF，4字节
} Frame_Data_t;                   // 总共56字节
#pragma pack(pop)

void printRawHex(const unsigned char* data, int len) {
    std::cout << "Raw data (hex): ";
    for (int i = 0; i < len; i++) {
        printf("%02X ", data[i]);
        if ((i + 1) % 16 == 0) std::cout << std::endl << "                ";
    }
    std::cout << std::endl;
}

int main() {
    const char* port = "/dev/rfcomm0";

    int fd = open(port, O_RDWR | O_NOCTTY);
    if (fd == -1) {
        std::cerr << "Failed to open " << port << std::endl;
        return -1;
    }

    struct termios options;
    tcgetattr(fd, &options);

    memset(&options, 0, sizeof(options));
    cfsetispeed(&options, B921600);
    cfsetospeed(&options, B921600);

    options.c_cflag |= (CLOCAL | CREAD);
    options.c_cflag &= ~PARENB;
    options.c_cflag &= ~CSTOPB;
    options.c_cflag &= ~CSIZE;
    options.c_cflag |= CS8;
    
    // 设置为原始二进制模式
    options.c_lflag &= ~(ICANON | ECHO | ECHOE | ISIG);
    options.c_iflag &= ~(ISTRIP | INLCR | IGNCR | ICRNL | IXON | IXOFF | IXANY);
    options.c_oflag &= ~OPOST;

    options.c_cc[VMIN] = 0;   // 非阻塞
    options.c_cc[VTIME] = 10; // 1秒超时

    if (tcsetattr(fd, TCSANOW, &options) != 0) {
        std::cerr << "Error setting serial port attributes" << std::endl;
        close(fd);
        return -1;
    }

    tcflush(fd, TCIOFLUSH);
    
    const int FRAME_SIZE = sizeof(Frame_Data_t);
    std::cout << "Frame size: " << FRAME_SIZE << " bytes (56 bytes expected)" << std::endl;
    std::cout << "Waiting for frame data..." << std::endl;

    unsigned char buffer[512];
    int buffer_len = 0;
    int frame_count = 0;
    
    while (true) {
        // 读取数据到缓冲区
        int n = read(fd, buffer + buffer_len, sizeof(buffer) - buffer_len);
        
        if (n > 0) {
            buffer_len += n;
            std::cout << "Received " << n << " bytes, buffer total: " << buffer_len << std::endl;
            
            // 在缓冲区中查找完整帧
            while (buffer_len >= FRAME_SIZE) {
                int frame_start = -1;
                
                // 查找帧头 0xFF (作为int32，小端序为 FF 00 00 00)
                for (int i = 0; i <= buffer_len - FRAME_SIZE; i++) {
                    // 检查是否为帧头 0xFF (小端序)
                    if (buffer[i] == 0xFF && buffer[i+1] == 0x00 && 
                        buffer[i+2] == 0x00 && buffer[i+3] == 0x00) {
                        
                        // 检查对应位置的帧尾是否为 0xEF (小端序)
                        int tail_pos = i + FRAME_SIZE - 4;
                        if (tail_pos + 3 < buffer_len &&
                            buffer[tail_pos] == 0xEF && buffer[tail_pos+1] == 0x00 && 
                            buffer[tail_pos+2] == 0x00 && buffer[tail_pos+3] == 0x00) {
                            frame_start = i;
                            break;
                        }
                    }
                }
                
                if (frame_start != -1) {
                    // 找到完整帧
                    Frame_Data_t* frame = (Frame_Data_t*)(buffer + frame_start);
                    
                    std::cout << "\n========== Frame " << ++frame_count << " ==========" << std::endl;
                    
                    // 显示原始十六进制数据（前20字节用于调试）
                    printRawHex(buffer + frame_start, (FRAME_SIZE > 20 ? 20 : FRAME_SIZE));
                    
                    // 验证帧头帧尾
                    std::cout << "Frame Head: 0x" << std::hex << frame->frame_head << std::dec;
                    if (frame->frame_head == 0xFF) {
                        std::cout << " ✓" << std::endl;
                    } else {
                        std::cout << " ✗ (expected 0xFF, got 0x" << std::hex << frame->frame_head << std::dec << ")" << std::endl;
                    }
                    
                    std::cout << "Frame Tail: 0x" << std::hex << frame->frame_tail << std::dec;
                    if (frame->frame_tail == 0xEF) {
                        std::cout << " ✓" << std::endl;
                    } else {
                        std::cout << " ✗ (expected 0xEF, got 0x" << std::hex << frame->frame_tail << std::dec << ")" << std::endl;
                    }
                    
                    // 输出坐标数据
                    std::cout << "\nCoordinates Data:" << std::endl;
                    for (int i = 0; i < 4; i++) {
                        std::cout << "Point " << i << ": ("
                                  << frame->coordinates_data[i][0] << ", "
                                  << frame->coordinates_data[i][1] << ", "
                                  << frame->coordinates_data[i][2] << ")" << std::endl;
                    }
                    std::cout << "==============================\n" << std::endl;
                    
                    // 移除已处理的帧
                    int bytes_to_remove = frame_start + FRAME_SIZE;
                    memmove(buffer, buffer + bytes_to_remove, buffer_len - bytes_to_remove);
                    buffer_len -= bytes_to_remove;
                    
                } else {
                    // 没找到完整帧
                    if (buffer_len > FRAME_SIZE + 10) {
                        // 如果缓冲区有足够数据但没找到帧，移除第一个字节继续查找
                        std::cout << "Frame not found, shifting buffer..." << std::endl;
                        memmove(buffer, buffer + 1, buffer_len - 1);
                        buffer_len--;
                    } else {
                        // 等待更多数据
                        break;
                    }
                }
            }
            
            // 防止缓冲区溢出
            if (buffer_len >= sizeof(buffer) - 100) {
                std::cout << "Buffer overflow protection: clearing buffer" << std::endl;
                buffer_len = 0;
            }
            
        } else if (n < 0) {
            std::cerr << "Read error" << std::endl;
            break;
        }
    }

    close(fd);
    return 0;
}