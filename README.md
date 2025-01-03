# ML-CPP Target Recognition System

A C++ implementation of a target recognition system that combines image and trajectory features for robust target classification.

## Features

- Dual-model fusion recognition system
- Real-time target tracking and classification
- Image-based target recognition using ResNet18
- Trajectory-based target recognition using custom features
- Evidence theory based fusion algorithm
- Multi-target management support


## Dependencies

- C++17 or higher
- CMake 3.10 or higher
- LibTorch (PyTorch C++ API)
- OpenCV 4.x
- xtensor
- xtensor-xtl
- CUDA (optional, for GPU support)

## Build Instructions

1. Install dependencies:
```bash
# Ubuntu/Debian
sudo apt-get update
sudo apt-get install build-essential cmake
sudo apt-get install python3-pip

mkdir third_party
# Download and setup LibTorch
cd third_party
wget https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-latest.zip
unzip libtorch-cxx11-abi-shared-with-deps-latest.zip

# Install OpenCV from the repository
sudo apt-get update
sudo apt-get install libopencv-dev

# Download and install xtensor and xtl
cd third_party
# Clone xtl
git clone https://github.com/xtensor-stack/xtl.git
cd xtl
mkdir build && cd build
cmake ..
sudo make install
cd ../..

# Clone xtensor
git clone https://github.com/xtensor-stack/xtensor.git
cd xtensor
mkdir build && cd build
cmake ..
sudo make install
cd ../..
```

2. Download models:
```bash
cd models
python3 download_resnet18.py
python3 download_imagenet_labels.py
```

3. Build the project:
```bash
mkdir build && cd build
cmake ..
make -j$(nproc)
```

## Usage Example

```cpp
// Initialize the prediction system
PredictionSystem system(
    "models/figure_model.pt",      // Image recognition model
    "models/trace_model.pt",       // Trajectory recognition model
    "models/trace_mean.npy",       // Feature normalization parameters
    "models/trace_scale.npy",
    5,                            // Trace smooth window
    0.04,                         // Target delta time
    20,                           // Base window size
    21,                           // Cache length
    DeviceType::CPU               // Device type (CPU/CUDA)
);

// Add a target
int target_id = 1;
system.add_target(target_id);

// Update target information
system.update_info_for_target_trace(
    target_id,
    obs_x, obs_y, obs_z,           // Observation position
    filter_p_x, filter_p_y, filter_p_z,  // Filtered position
    filter_v_x, filter_v_y, filter_v_z,  // Filtered velocity
    filter_a_x, filter_a_y, filter_a_z   // Filtered acceleration
);

// Update target image
system.update_info_for_target_figure(target_id, image_data);

// Get recognition results
std::vector<float> trace_probs, figure_probs;
system.trace_model_recognition(target_id, trace_probs);
system.figure_model_recognition(target_id, figure_probs);

// Get fusion result
int predicted_class = system.get_fusion_target_recognition(
    target_id, trace_probs, figure_probs
);
```

## Testing

The project includes comprehensive test suites for each component:

- `data_processor_test`: Tests for image and trace data preprocessing
- `feature_store_test`: Tests for feature storage and computation
- `model_wrapper_test`: Tests for model loading and inference
- `prediction_system_test`: Tests for the complete prediction pipeline
- `target_manager_test`: Tests for target management functionality

Run tests:
```bash
cd build
./ml_predictor_node  # Run data processor tests
```

## Performance Optimization

- Uses CUDA acceleration when available
- Efficient memory management with smart pointers
- Optimized tensor operations with LibTorch
- Batch processing support for multiple targets

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.
