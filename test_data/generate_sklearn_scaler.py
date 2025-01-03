import os
import numpy as np
from sklearn.preprocessing import StandardScaler

def print_array_info(name, arr):
    """Helper function to print detailed array information."""
    print(f"\n=== {name} ===")
    print(f"Type: {type(arr)}")
    print(f"DType: {arr.dtype}")
    print(f"Shape: {arr.shape}")
    print(f"Data:\n{arr}")
    if isinstance(arr, np.ndarray):
        print(f"Data (high precision):\n{arr.astype(np.float64)}")

def generate_test_data():
    """Generate test data and save scaler parameters."""
    print("\n=== Configuration ===")
    print(f"NumPy version: {np.__version__}")
    
    # Define test data with known values for easy verification
    data = np.array([
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
        [7.0, 8.0, 9.0]
    ], dtype=np.float64)
    print_array_info("Original Data", data)

    # Create and fit StandardScaler
    scaler = StandardScaler()
    scaler.fit(data)

    # Calculate and print statistics
    print("\n=== Detailed Statistics ===")
    print("Mean:", scaler.mean_)
    print("Scale:", scaler.scale_)
    print("Variance:", np.var(data, axis=0, ddof=1))

    # Transform using sklearn for reference
    transformed_data = scaler.transform(data)
    print("\n=== sklearn Transform Result ===")
    print(transformed_data)

    # Save parameters
    mean_path = "mean.npy"
    scale_path = "scale.npy"
    
    # Save mean and scale
    np.save(mean_path, scaler.mean_.astype(np.float64))
    np.save(scale_path, scaler.scale_.astype(np.float64))

    # Verify saved parameters
    loaded_mean = np.load(mean_path)
    loaded_scale = np.load(scale_path)
    print("\n=== Loaded Parameters ===")
    print("Loaded Mean:", loaded_mean)
    print("Loaded Scale:", loaded_scale)

    # Verify loaded parameters match original
    print("\n=== Parameter Verification ===")
    mean_diff = np.abs(scaler.mean_ - loaded_mean).max()
    scale_diff = np.abs(scaler.scale_ - loaded_scale).max()
    print(f"Max mean difference: {mean_diff}")
    print(f"Max scale difference: {scale_diff}")

    # Manual transform using loaded parameters
    # Use the formula: (X - mean) * scale for each column
    manual_transform = np.zeros_like(data)
    for j in range(data.shape[1]):  # 对每一列进行标准化
        manual_transform[:, j] = (data[:, j] - loaded_mean[j]) / loaded_scale[j]
    
    print("\n=== Manual Transform Steps ===")
    print("Original shape:", data.shape)
    print("Mean shape:", loaded_mean.shape)
    print("Scale shape:", loaded_scale.shape)
    print("Final Result:\n", manual_transform)

    # Verify results match sklearn
    print("\n=== Transform Verification ===")
    abs_diff = np.abs(transformed_data - manual_transform)
    print("Max absolute difference:", np.max(abs_diff))
    
    try:
        np.testing.assert_array_almost_equal(transformed_data, manual_transform, decimal=10)
        print("Validation Passed: Results match with tolerance=1e-10")
    except AssertionError as e:
        print("Validation Failed!")
        print(str(e))

    # Save test data and expected results for C++ validation
    np.save("test_data.npy", data.astype(np.float64))
    np.save("expected_transform.npy", transformed_data.astype(np.float64))

    # Final verification
    print("\n=== Final Verification ===")
    final_data = np.load("test_data.npy")
    final_expected = np.load("expected_transform.npy")
    final_mean = np.load(mean_path)
    final_scale = np.load(scale_path)
    
    print("Saved test data shape:", final_data.shape)
    print("Saved expected result shape:", final_expected.shape)
    print("Saved mean:", final_mean)
    print("Saved scale:", final_scale)

if __name__ == "__main__":
    generate_test_data()