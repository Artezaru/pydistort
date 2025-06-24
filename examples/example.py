from pydistort import distort_image, Cv2Distortion
import numpy
import cv2
import os

path = os.path.join(os.path.dirname(__file__), 'ORI.jpg')
image = cv2.imread(path, cv2.IMREAD_COLOR)

if image is None:
    raise FileNotFoundError(f"Image not found at {path}")

# Define the intrinsic camera matrix
height, width = image.shape[:2]


K = numpy.array([[10000.0, 0.0, width / 2],
                 [0.0, 10000.0, height / 2],
                 [0.0, 0.0, 1.0]], dtype=numpy.float64)

# Define the distortion model
distortion = Cv2Distortion([0.1, 0.2, 0.3, 0.4, 0.5])
distortion.parameters *= 10

# Distort the image
distorted_image = distort_image(image, K, distortion)

# Save the distorted image
output_path = os.path.join(os.path.dirname(__file__), 'ORI_distorted.jpg')
cv2.imwrite(output_path, distorted_image)

# Display the original and distorted images
cv2.imshow('Original Image', image)
cv2.imshow('Distorted Image', distorted_image)
cv2.waitKey(0)