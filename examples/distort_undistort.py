from pydistort import distort_image, Cv2Distortion
from pydistort.objects import Intrinsic
import cv2
import os
import numpy
import matplotlib.pyplot as plt

def CV2_DISTORTION_1():
    """Create a Cv2Distortion object with specified parameters."""
    distortion = Cv2Distortion(Nparams=5)
    # Set distortion coefficients
    distortion.k1 = 0.017083945091492785
    distortion.k2 = -0.1093719257958107
    distortion.p1 = 0.04280641095874525
    distortion.p2 = -0.11948575638043393
    distortion.k3 = 0.0908833886027441
    
    return distortion


image = cv2.imread(os.path.join(os.path.dirname(__file__), 'ORI.jpg'))
print(f'Image shape: {image.shape}, dtype: {image.dtype}')
height, width = image.shape[:2]

distortion = CV2_DISTORTION_1()

fx = 10000.0
fy = 9500.0
cx = width / 2.0
cy = height / 2.0
intrinsic = Intrinsic()
intrinsic.intrinsic_matrix = numpy.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=numpy.float64)

image_distorted_method_undistort = distort_image(image, intrinsic.K, distortion, method='undistort')
image_distorted_method_distort = distort_image(image, intrinsic.K, distortion, method='distort')
image_distorted_method_distort = numpy.round(image_distorted_method_distort).astype(numpy.uint8)

print(f'Distorted image shape: {image_distorted_method_distort.shape}, dtype: {image_distorted_method_distort.dtype}')
print(f'Distorted image (undistort method) shape: {image_distorted_method_undistort.shape}, dtype: {image_distorted_method_undistort.dtype}')

print(image_distorted_method_distort)

pixel_points = numpy.indices((height, width), dtype=numpy.float64) # shape (2, H, W)
pixel_points = pixel_points.reshape(2, -1).T  # shape (H*W, 2) WARNING: [H, W -> Y, X]
pixel_points = pixel_points[:, [1, 0]]  # Swap to [X, Y] format
normalized_points, _, _ = intrinsic._inverse_transform(pixel_points, dx=False, dp=False)
distorted_points, _, _ = distortion._transform(normalized_points)
pixel_points_distorted, _, _ = intrinsic._transform(distorted_points)

# Display the original and distorted images
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.scatter(pixel_points[:, 0], pixel_points[:, 1], s=1, c='red', label='Distorted Points')
plt.scatter(pixel_points_distorted[:, 0], pixel_points_distorted[:, 1], s=1, c='blue', label='Original Points')
plt.title('Distorted vs Original Points')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.legend()
plt.axis('equal')
plt.show()

cv2.imshow('Original Image', image)
cv2.imshow('Distorted Image (undistort method)', image_distorted_method_undistort)
cv2.imshow('Distorted Image (distort method)', image_distorted_method_distort)
cv2.waitKey(0)

