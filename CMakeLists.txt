cmake_minimum_required(VERSION 3.10)
project(ml_predictor)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_BUILD_TYPE Release)

# Find OpenCV
find_package(OpenCV REQUIRED)

# Add LibTorch Support
list(APPEND CMAKE_PREFIX_PATH ${CMAKE_CURRENT_SOURCE_DIR}/third_party/libtorch)
find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

# xtensor and xtl
add_subdirectory(third_party/xtl)
include_directories(third_party/xtl/include)
add_subdirectory(third_party/xtensor)
include_directories(third_party/xtensor/include)

# Define the executable
add_executable(ml_predictor_node 
    src/prediction_system_test.cpp
    modules/feature_store/batch_vector.cpp 
    modules/feature_store/feature_store.cpp 
    modules/preprocessor/data_preprocessor.cpp
    modules/target_manager/model_wrapper.cpp 
    modules/target_manager/target_manager.cpp 
    modules/target_manager/prediction_system.cpp 
)

# Link libraries
target_link_libraries(ml_predictor_node PRIVATE
    ${TORCH_LIBRARIES}    # Link LibTorch
    ${OpenCV_LIBS}        # Link OpenCV (automatically includes necessary components)
    xtensor               # xtensor
    xtl                   # xtl
    pthread               # Linux threading library
)

# Configure RPATH for runtime library discovery
if (UNIX)
    set_target_properties(ml_predictor_node PROPERTIES
        INSTALL_RPATH_USE_LINK_PATH TRUE
        BUILD_WITH_INSTALL_RPATH TRUE
        INSTALL_RPATH "${CMAKE_CURRENT_SOURCE_DIR}/third_party/libtorch/lib"
    )
endif()