cmake_minimum_required(VERSION 3.16)
project(WDR_transform LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ========================
# 依赖查找
# ========================
find_package(OpenCV REQUIRED)

# ========================
# 源文件
# ========================
add_executable(transform
    transform.cpp
)

# ========================
# 头文件路径
# ========================
target_include_directories(transform PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${OpenCV_INCLUDE_DIRS}
)

# ========================
# 链接库
# ========================
target_link_libraries(transform PRIVATE
    ${OpenCV_LIBS}
)

# ========================
# 编译选项
# ========================
target_compile_options(transform PRIVATE 
    -Wall 
    -O2
)

message(STATUS "========================================")
message(STATUS "Project: ${PROJECT_NAME}")
message(STATUS "OpenCV version: ${OpenCV_VERSION}")
message(STATUS "========================================")