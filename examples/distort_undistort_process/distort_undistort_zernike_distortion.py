
# %% 
# ===============================================================================
# ======================= IMPORTATIONS ==========================================
# ===============================================================================

import numpy 
import matplotlib.pyplot as plt
import cv2
import os
import csv
import copy

from pydistort import ZernikeDistortion, distort_image, undistort_image











# %% 
# ===============================================================================
# ======================= Readers ===============================================
# ===============================================================================
def read_array1D(file_path):
    """
    Reads a 1D array from a text file.
    """
    values = []
    with open(file_path, newline='') as f:
        reader = csv.reader(f, delimiter=';')
        for row in reader:
            try:
                float_values = [float(x) for x in row if x.strip()]
                values.extend(float_values)
            except ValueError:
                continue  # Skip malformed or empty lines
    return numpy.array(values, dtype=numpy.float64)

def write_array1D(file_path, array):
    """
    Writes a 1D array to a text file.
    """
    with open(file_path, 'w', newline='') as f:
        writer = csv.writer(f, delimiter=';')
        N = 30  # optional: number of values per line
        for i in range(0, len(array), N):
            row = [f"{x:.18e}" for x in array[i:i+N]]
            writer.writerow(row)

def read_array2D(file_path):
    """
    Reads a 2D array from a text file.
    """
    values = []
    with open(file_path, newline='') as f:
        reader = csv.reader(f, delimiter=';')
        for row in reader:
            try:
                float_values = [float(x) for x in row if x.strip()]
                values.append(float_values)
            except ValueError:
                continue  # Skip malformed or empty lines
    return numpy.array(values, dtype=numpy.float64)

def write_array2D(file_path, array):
    """
    Writes a 2D array to a text file.
    """
    with open(file_path, 'w', newline='') as f:
        writer = csv.writer(f, delimiter=';')
        for row in array:
            formatted_row = [f"{x:.18e}" for x in row]
            writer.writerow(formatted_row)










# %%
# ===============================================================================
# ============== CREATE IMAGES AND DISTORTION ===================================
# ===============================================================================

parameters_file = os.path.join(os.path.dirname(__file__), "files", "zernike_parameters.txt")
parameters = read_array1D(parameters_file)

image_file = os.path.join(os.path.dirname(__file__), "files", "speckle.png")
distorted_image_file = lambda interp: os.path.join(os.path.dirname(__file__), "files", f"distorted_speckle_{interp}.png")
undistorted_image_file = lambda interp: os.path.join(os.path.dirname(__file__), "files", f"undistorted_speckle_{interp}.png")
error_image_file = lambda interp: os.path.join(os.path.dirname(__file__), "files", f"error_speckle_{interp}.png")
cropped_error_image_file = lambda interp: os.path.join(os.path.dirname(__file__), "files", f"cropped_error_speckle_{interp}.png")

max_error = 10  # in gray levels

# Read the image
image = cv2.imread(image_file, cv2.IMREAD_GRAYSCALE)
image = image.astype(numpy.uint8)
image_height, image_width = image.shape[:2]

# Create the distortion object
real_distortion = ZernikeDistortion(parameters=parameters)
print(f"real distortion object:\n{real_distortion}\n")

# Rescale the distortion model to the image size
radius = numpy.sqrt((image_width / 2) ** 2 + (image_height / 2) ** 2)
center = (image_width / 2, image_height / 2)
real_distortion.radius = radius
real_distortion.center = center

# Create pixel points in the image
pixel_points = numpy.indices((image_height, image_width), dtype=numpy.float64) # shape (2, H, W)
pixel_points = pixel_points.reshape(2, -1).T  # shape (H*W, 2) WARNING: [H, W -> Y, X]
pixel_points = pixel_points[:, [1, 0]]  # Swap to [X, Y] format














# %%
# ===============================================================================
# ======================= Distort / Undistort ===================================
# ===============================================================================

method = "undistort"
interpolation = "spline3" 

# Distort the image
distorted_image = distort_image(src=image, intrinsic=None, distortion=real_distortion, method=method, interpolation=interpolation)
print(numpy.min(distorted_image), numpy.max(distorted_image), numpy.mean(distorted_image), numpy.std(distorted_image))
distorted_image[distorted_image < 0] = 0.0
distorted_image[distorted_image > 255] = 255.0
distorted_image = numpy.round(distorted_image).astype(numpy.uint8)

# Distorted pixel points
result = real_distortion.distort(pixel_points)
distorted_pixel_points = result.transformed_points

# Undistort the image
undistorted_image = undistort_image(src=distorted_image, intrinsic=None, distortion=real_distortion, interpolation=interpolation)
print(numpy.min(undistorted_image), numpy.max(undistorted_image), numpy.mean(undistorted_image), numpy.std(undistorted_image))
undistorted_image[undistorted_image < 0] = 0.0
undistorted_image[undistorted_image > 255] = 255.0
undistorted_image = numpy.round(undistorted_image).astype(numpy.uint8)

# Undistorted pixel points
result = real_distortion.undistort(distorted_pixel_points)
undistorted_pixel_points = result.transformed_points

# Save the distorted and undistorted images
cv2.imwrite(distorted_image_file(interpolation), distorted_image)
cv2.imwrite(undistorted_image_file(interpolation), undistorted_image)

# Calculate the error between the original and undistorted images
error_image = cv2.absdiff(image, undistorted_image)
print(error_image.shape, error_image.dtype)
print(f"Error image min: {numpy.min(error_image)}, max: {numpy.max(error_image)}")
print(f"Error image mean: {numpy.mean(error_image)}, std: {numpy.std(error_image)}")

# Save the error images
cv2.imwrite(error_image_file(interpolation), error_image)

# Rescale the error image to [0, max_error]
cropped_error_image = numpy.clip(error_image, 0, max_error)
cropped_error_image = (cropped_error_image / max_error * 255).astype(numpy.uint8)

# Save the cropped error image
cv2.imwrite(cropped_error_image_file(interpolation), cropped_error_image)

# Compute the mean and standard deviation of the error in points
error_points = numpy.abs(pixel_points - undistorted_pixel_points)

# Display the error image
max_error = 10 # in gray levels

plt.figure(figsize=(6,6))
plt.subplot(1, 1, 1)
plt.imshow(error_image, cmap='gray', vmin=0, vmax=max_error)
plt.colorbar(label='Error')
plt.title(f"Error Image with {interpolation} interpolation")
plt.axis('off')
plt.tight_layout()

plt.figure(figsize=(6,6))
plt.subplot(1, 1, 1)
plt.scatter(pixel_points[:, 0], pixel_points[:, 1], c=numpy.linalg.norm(error_points, axis=1), cmap='jet', s=1)
plt.colorbar(label='Error Norm')
plt.title(f"Error Points with {interpolation} interpolation")
plt.axis('equal')
plt.tight_layout()

plt.show()