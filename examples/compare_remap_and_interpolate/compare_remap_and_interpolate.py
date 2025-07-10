
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
import scipy

from pydistort import ZernikeDistortion











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
distorted_image_file = os.path.join(os.path.dirname(__file__), "files", "distorted_speckle.png")
undistorted_image_file = os.path.join(os.path.dirname(__file__), "files", "undistorted_speckle.png")
error_image_file = os.path.join(os.path.dirname(__file__), "files", "error_speckle.png")

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
# ======================= UNDISTORT THE POINTS ==================================
# ===============================================================================
# ! WARNING - IS RELATIVE TO THE PIXEL POINTS
result = real_distortion.undistort(pixel_points)
undistorted_points = result.transformed_points

# ! WARNING - IS RELATIVE TO THE PIXEL POINTS
result = real_distortion.distort(pixel_points)
distorted_points = result.transformed_points












# %%
# ===============================================================================
# ======= CREATE THE DISTORTED IMAGES FOR DIFFERENT INTERP LINEAR ===============
# ===============================================================================



# ! CV2 REMAP
# Create the distorted image using remap cv2 (linear interpolation)
src = image.copy()  # Copy the original image to avoid modifying it

u_points = undistorted_points[:, [1, 0]]  # Switch to [Y, X] format for cv2.remap, shape (Npoints, 2) [X', Y'] -> shape (Npoints, 2) [Y', X']
u_points = u_points.T.reshape(2, image_height, image_width)  # shape (Npoints, 2) [Y', X'] -> shape (2, H, W) [Y', X']

map_x = u_points[1, :, :]  # X coordinates, shape (H, W)
map_y = u_points[0, :, :]  # Y coordinates, shape (H, W)

cv2_remap_distorted_image = cv2.remap(src, map_x.astype(numpy.float32), map_y.astype(numpy.float32), interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))

# Create the undistorted image using remap cv2 (linear interpolation)
src = cv2_remap_distorted_image.copy()  # Copy the distorted image to avoid modifying it

d_points = distorted_points[:, [1, 0]]  # Switch to [Y, X] format for cv2.remap, shape (Npoints, 2) [X', Y'] -> shape (Npoints, 2) [Y', X']
d_points = d_points.T.reshape(2, image_height, image_width)  # shape (Npoints, 2) [Y', X'] -> shape (2, H, W) [Y', X']

map_x = d_points[1, :, :]  # X coordinates, shape (H, W)
map_y = d_points[0, :, :]  # Y coordinates, shape (H, W)

cv2_remap_image = cv2.remap(src, map_x.astype(numpy.float32), map_y.astype(numpy.float32), interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))





# ! Scipy LINEAR ND Interpolation
# Create the distorted image using scipy (linear interpolation)
src = image.copy()  # Copy the original image to avoid modifying it
values = src.reshape(image_height, image_width, -1)  # shape (H, W, 1 * [C] * [D])
values = values.reshape(-1, values.shape[-1])  # Reshape to (H*W, C) for interpolation

u_points = undistorted_points[:, [1, 0]]  # Switch to [Y, X] format for scipy, shape (Npoints, 2) [X', Y'] -> shape (Npoints, 2) [Y', X']

scipy_linearNdinterpolator_distorted_image = numpy.zeros_like(values, dtype=numpy.float64)  # Initialize the distorted image array
image_points = pixel_points[:, [1, 0]] # Switch to [Y, X] format for scipy.interpolate, shape (Npoints, 2) [X', Y'] -> shape (Npoints, 2) [Y', X']

for i in range(values.shape[-1]):
    mask = numpy.isfinite(values[:, i])
    image_points_filtered = image_points[mask, :]
    values_filtered = values[mask, i]

    interp = scipy.interpolate.LinearNDInterpolator(image_points_filtered, values_filtered, fill_value=0)

    result = interp(u_points)
    scipy_linearNdinterpolator_distorted_image[:, i] = result

scipy_linearNdinterpolator_distorted_image = scipy_linearNdinterpolator_distorted_image.reshape(image_height, image_width, -1)  # Reshape back to (H, W, C)
scipy_linearNdinterpolator_distorted_image = scipy_linearNdinterpolator_distorted_image.reshape(image_height, image_width, *src.shape[2:])  # Reshape to (H, W, C) or (H, W, C, D) if D > 1

# Create the undistorted image using scipy (linear interpolation)
src = scipy_linearNdinterpolator_distorted_image.copy()  # Copy the distorted image to avoid modifying it
values = src.reshape(image_height, image_width, -1)  # shape (H, W, 1 * [C] * [D])
values = values.reshape(-1, values.shape[-1])  # Reshape to (H*W, C) for interpolation

d_points = distorted_points[:, [1, 0]]  # Switch to [Y, X] format for scipy, shape (Npoints, 2) [X', Y'] -> shape (Npoints, 2) [Y', X']

scipy_linearNdinterpolator_image = numpy.zeros_like(values, dtype=numpy.float64)  # Initialize the distorted image array
image_points = pixel_points[:, [1, 0]] # Switch to [Y, X] format for scipy.interpolate, shape (Npoints, 2) [X', Y'] -> shape (Npoints, 2) [Y', X']

for i in range(values.shape[-1]):
    mask = numpy.isfinite(values[:, i])
    image_points_filtered = image_points[mask, :]
    values_filtered = values[mask, i]

    interp = scipy.interpolate.LinearNDInterpolator(image_points_filtered, values_filtered, fill_value=0)

    result = interp(d_points)
    scipy_linearNdinterpolator_image[:, i] = result

scipy_linearNdinterpolator_image = scipy_linearNdinterpolator_image.reshape(image_height, image_width, -1)  # Reshape back to (H, W, C)
scipy_linearNdinterpolator_image = scipy_linearNdinterpolator_image.reshape(image_height, image_width, *src.shape[2:])  # Reshape to (H, W, C) or (H, W, C, D) if D > 1










# ! My INTERPOLATION
def my_interpolation_lineaire(image, x, y):
    h, w = image.shape[:2]

    # Vérifie si (x, y) est hors des limites
    if x < 0 or y < 0 or x >= w - 1 or y >= h - 1:
        return 0

    x0 = int(numpy.floor(x))
    x1 = x0 + 1
    y0 = int(numpy.floor(y))
    y1 = y0 + 1

    dx = x - x0
    dy = y - y0

    # Interpolation bilinéaire
    top = (1 - dx) * image[y0, x0] + dx * image[y0, x1]
    bottom = (1 - dx) * image[y1, x0] + dx * image[y1, x1]
    value = (1 - dy) * top + dy * bottom

    return value

# Create the distorted image using my linear interpolation
src = image.copy()  # Copy the original image to avoid modifying it

u_points = undistorted_points[:, [1, 0]]  # Switch to [Y, X] format for my interpolation, shape (Npoints, 2) [X', Y'] -> shape (Npoints, 2) [Y', X']
u_points = u_points.T.reshape(2, image_height, image_width)  # shape (Npoints, 2) [Y', X'] -> shape (2, H, W) [Y', X']

my_distorted_image = numpy.zeros_like(src, dtype=numpy.float64)  # Initialize the distorted image array

for i in range(image_height):
    for j in range(image_width):
        x = u_points[1, i, j]  # X coordinate
        y = u_points[0, i, j]  # Y coordinate
        my_distorted_image[i, j] = my_interpolation_lineaire(src, x, y)












# ! Scipy RegularGridInterpolator
# Create the distorted image using scipy (linear interpolation)
src = image.copy()  # Copy the original image to avoid modifying it
values = src.reshape(image_height, image_width, -1)  # shape (H, W, 1 * [C] * [D])

grid_y = numpy.arange(image_height)  # Create a grid for Y coordinates
grid_x = numpy.arange(image_width)  # Create a grid for X coordinates

u_points = undistorted_points[:, [1, 0]]  # Switch to [Y, X] format for scipy, shape (Npoints, 2) [X', Y'] -> shape (Npoints, 2) [Y', X']

scipy_regulargridinterpolator_distorted_image = numpy.zeros_like(values, dtype=numpy.float64)  # Initialize the distorted image array

for i in range(values.shape[-1]):

    interp = scipy.interpolate.RegularGridInterpolator(
        (grid_y, grid_x), values[:, :, i], method='linear', bounds_error=False, fill_value=0.0
    )

    result = interp(u_points)  # Transpose u_points to (2, Npoints) for interpolation
    result = result.reshape(image_height, image_width)  # Reshape to (H, W)
    scipy_regulargridinterpolator_distorted_image[:, :, i] = result

scipy_regulargridinterpolator_distorted_image = scipy_regulargridinterpolator_distorted_image.reshape(image_height, image_width, *src.shape[2:])  # Reshape to (H, W, C) or (H, W, C, D) if D > 1








# %%
# ===============================================================================
# ======================= DISPLAY TYPES =========================================
# ===============================================================================

print(f"Image shape: {image.shape}, dtype: {image.dtype}")
print(f"CV2 remap distorted image shape: {cv2_remap_distorted_image.shape}, dtype: {cv2_remap_distorted_image.dtype}")
print(f"SciPy LinearNDInterpolator distorted image shape: {scipy_linearNdinterpolator_distorted_image.shape}, dtype: {scipy_linearNdinterpolator_distorted_image.dtype}")
print(f"My Linear Interpolation distorted image shape: {my_distorted_image.shape}, dtype: {my_distorted_image.dtype}")
print(f"SciPy RegularGridInterpolator distorted image shape: {scipy_regulargridinterpolator_distorted_image.shape}, dtype: {scipy_regulargridinterpolator_distorted_image.dtype}")

print(f"CV2 remap image shape: {cv2_remap_image.shape}, dtype: {cv2_remap_image.dtype}")
print(f"SciPy LinearNDInterpolator image shape: {scipy_linearNdinterpolator_image.shape}, dtype: {scipy_linearNdinterpolator_image.dtype}")







# %%
# ===============================================================================
# ======================= ROUND THE IMAGES ======================================
# ===============================================================================

cv2_remap_distorted_image = numpy.round(cv2_remap_distorted_image).astype(numpy.uint8)  # Round and convert to uint8
scipy_linearNdinterpolator_distorted_image = numpy.round(scipy_linearNdinterpolator_distorted_image).astype(numpy.uint8)  # Round and convert to uint8
my_distorted_image = numpy.round(my_distorted_image).astype(numpy.uint8)  # Round and convert to uint8
scipy_regulargridinterpolator_distorted_image = numpy.round(scipy_regulargridinterpolator_distorted_image).astype(numpy.uint8)  # Round and convert to uint8

cv2_remap_image = numpy.round(cv2_remap_image).astype(numpy.uint8)  # Round and convert to uint8
scipy_linearNdinterpolator_image = numpy.round(scipy_linearNdinterpolator_image).astype(numpy.uint8)  # Round and convert to uint8






# %%
# ===============================================================================
# ======================= DISPLAY IMAGES ========================================
# ===============================================================================

error_max = 10

plt.figure(figsize=(25, 5))
plt.subplot(1, 5, 1)
plt.imshow(image, cmap='gray')
plt.title("Original Image")
plt.axis('off')
plt.subplot(1, 5, 2)
plt.imshow(cv2_remap_distorted_image, cmap='gray')
plt.title("cv2.remap Distorted Image")
plt.axis('off')
plt.subplot(1, 5, 3)
plt.imshow(scipy_linearNdinterpolator_distorted_image, cmap='gray')
plt.title("SciPy LinearNDInterpolator Distorted Image")
plt.axis('off')
plt.subplot(1, 5, 4)
plt.imshow(my_distorted_image, cmap='gray')
plt.title("My Linear Interpolation Distorted Image")
plt.axis('off')
plt.subplot(1, 5, 5)
plt.imshow(scipy_regulargridinterpolator_distorted_image, cmap='gray')
plt.title("SciPy RegularGridInterpolator Distorted Image")
plt.axis('off')
plt.tight_layout()
plt.show()




# Display the difference images with CV2 remap
plt.figure(figsize=(5, 15))
plt.subplot(3, 1, 1)
plt.imshow(numpy.abs(cv2_remap_distorted_image.astype(numpy.float64) - scipy_linearNdinterpolator_distorted_image.astype(numpy.float64)), cmap='gray', vmin=0, vmax=error_max)
plt.colorbar()
plt.title("Difference between cv2.remap and SciPy LinearNDInterpolator Distorted Images")
plt.axis('off')
plt.subplot(3, 1, 2)
plt.imshow(numpy.abs(cv2_remap_distorted_image.astype(numpy.float64) - my_distorted_image.astype(numpy.float64)), cmap='gray', vmin=0, vmax=error_max)
plt.colorbar()
plt.title("Difference between cv2.remap and My Linear Interpolation Distorted Images")
plt.axis('off')
plt.subplot(3, 1, 3)
plt.imshow(numpy.abs(cv2_remap_distorted_image.astype(numpy.float64) - scipy_regulargridinterpolator_distorted_image.astype(numpy.float64)), cmap='gray', vmin=0, vmax=error_max)
plt.colorbar()
plt.title("Difference between cv2.remap and SciPy RegularGridInterpolator Distorted Images")
plt.axis('off')
plt.tight_layout()
plt.show()







# %%
# ===============================================================================
# ======================= DISPLAY IMAGES ========================================
# ===============================================================================

error_max = 10

plt.figure(figsize=(15, 5))
plt.subplot(1, 3, 1)
plt.imshow(image, cmap='gray')
plt.title("Original Image")
plt.axis('off')
plt.subplot(1, 3, 2)
plt.imshow(cv2_remap_image, cmap='gray')
plt.title("cv2.remap Image")
plt.axis('off')
plt.subplot(1, 3, 3)
plt.imshow(scipy_linearNdinterpolator_image, cmap='gray')
plt.title("SciPy LinearNDInterpolator Image")
plt.axis('off')
plt.tight_layout()
plt.show()


# Display the difference image from original
plt.figure(figsize=(5, 15))
plt.subplot(3, 1, 1)
plt.imshow(numpy.abs(image.astype(numpy.float64) - cv2_remap_image.astype(numpy.float64)), cmap='gray', vmin=0, vmax=error_max)
plt.colorbar()
plt.title("Difference between Original and cv2.remap Image")
plt.axis('off')
plt.subplot(3, 1, 2)
plt.imshow(numpy.abs(image.astype(numpy.float64) - scipy_linearNdinterpolator_image.astype(numpy.float64)), cmap='gray', vmin=0, vmax=error_max)
plt.colorbar()
plt.title("Difference between Original and SciPy LinearNDInterpolator Image")
plt.axis('off')
plt.subplot(3, 1, 3)
plt.imshow(numpy.abs(cv2_remap_image.astype(numpy.float64) - scipy_linearNdinterpolator_image.astype(numpy.float64)), cmap='gray', vmin=0, vmax=error_max)
plt.colorbar()
plt.title("Difference between cv2.remap and SciPy LinearNDInterpolator Image")
plt.axis('off')
plt.tight_layout()
plt.show()































