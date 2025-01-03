#!/bin/bash

# 设置颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# 打印带颜色的信息函数
print_info() {
    echo -e "${GREEN}[INFO] $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}[WARNING] $1${NC}"
}

print_error() {
    echo -e "${RED}[ERROR] $1${NC}"
}

# 检查必要的工具
check_dependencies() {
    print_info "Checking dependencies..."
    
    # 检查cmake
    if ! command -v cmake &> /dev/null; then
        print_error "cmake not found. Please install cmake first."
        exit 1
    fi
    
    # 检查make
    if ! command -v make &> /dev/null; then
        print_error "make not found. Please install make first."
        exit 1
    fi
    
    # 检查g++
    if ! command -v g++ &> /dev/null; then
        print_error "g++ not found. Please install g++ first."
        exit 1
    fi
}

# 清理构建目录
clean_build() {
    if [ "$1" = "clean" ]; then
        print_info "Cleaning build directory..."
        rm -rf build
    fi
}

# 创建并进入构建目录
create_build_dir() {
    print_info "Creating build directory..."
    mkdir -p build
    cd build || exit 1
}

# 运行CMake配置
run_cmake() {
    print_info "Running CMake configuration..."
    cmake .. -DCMAKE_BUILD_TYPE=Release
    if [ $? -ne 0 ]; then
        print_error "CMake configuration failed!"
        exit 1
    fi
}

# 编译项目
build_project() {
    print_info "Building project..."
    # 获取CPU核心数，用于并行编译
    CORES=$(nproc 2>/dev/null || echo 4)
    make -j"$CORES"
    if [ $? -ne 0 ]; then
        print_error "Build failed!"
        exit 1
    fi
}

# 主函数
main() {
    print_info "Starting build process..."
    
    # 检查依赖
    check_dependencies
    
    # 清理构建（如果需要）
    clean_build "$1"
    
    # 创建并进入构建目录
    create_build_dir
    
    # 运行CMake
    run_cmake
    
    # 编译项目
    build_project
    
    print_info "Build completed successfully!"
}

# 运行主函数，传入命令行参数
main "$1" 