import numpy as np
import cv2

# Define intrinsic matrix as identity (3x3)
K = np.eye(3)

# No rotation (identity) and no translation (zero vector)
rvec = np.zeros(3)  # Rodrigues rotation vector
tvec = np.zeros(3)

# Define some 3D points in front of the camera
object_points = np.array([
    [0.0, 0.0, 1.0],
    [0.1, 0.0, 1.0],
    [0.0, 0.1, 1.0],
    [0.1, 0.1, 1.0],
    [-0.1, -0.1, 1.0]
], dtype=np.float32)

# Convert to shape (N, 1, 3) as expected by OpenCV
object_points = object_points.reshape(-1, 1, 3)

# Simulate distortion coefficients (k1, k2, p1, p2, k3)
# Use some arbitrary non-zero distortion
dist_coeffs = np.array([0.2, -0.3, 0.001, 0.001, 0.0], dtype=np.float32)

# Project the 3D points to 2D image points (with distortion)
image_points, _ = cv2.projectPoints(object_points, rvec, tvec, K, dist_coeffs)

# Distorted points in shape (N, 1, 2)
distorted_points = image_points

# Undistort the 2D points back to normalized coordinates
# (Note: undistortPoints returns normalized coordinates when no K is passed)
undistorted_norm_points = cv2.undistortPoints(distorted_points, K, dist_coeffs)

# Compute the original normalized image coordinates (before any distortion)
# Since pose is identity and K is identity, normalized image points are X/Z, Y/Z
normalized_points = object_points[:, 0, :2] / object_points[:, 0, 2:]

# Compare the undistorted normalized points with the original normalized points
print("Original normalized points:")
print(normalized_points)

print("\nRecovered normalized points from undistort:")
print(undistorted_norm_points[:, 0, :])

# Optional: compute error
errors = np.linalg.norm(normalized_points - undistorted_norm_points[:, 0, :], axis=1)
print("\nReprojection error per point:")
print(errors)
print(f"\nMean reprojection error: {errors.mean()}")
