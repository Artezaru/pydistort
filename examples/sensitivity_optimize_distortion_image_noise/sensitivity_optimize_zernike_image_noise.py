
# %% 
# ===============================================================================
# ======================= IMPORTATIONS ==========================================
# ===============================================================================

STOP = True
import sys
if STOP:
    sys.exit("This example is bloqued for now to avoid erasing current files. Please remove the STOP variable to run it.")


import numpy 
import matplotlib.pyplot as plt
import cv2
import os
import csv
import copy
import skimage

from pydistort import ZernikeDistortion, cv2_distort_image



DISPLAY_INTERMEDIATE = False




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
# ============== CREATE IMAGES AND REAL DISTORTION ==============================
# ===============================================================================

parameters_file = os.path.join(os.path.dirname(__file__), "files", "zernike_parameters.txt")
parameters = read_array1D(parameters_file)

image_file = os.path.join(os.path.dirname(__file__), "files", "speckle.png")
distorted_image_file = os.path.join(os.path.dirname(__file__), "files", "distorted_speckle.png")

# Read the image
image = cv2.imread(image_file, cv2.IMREAD_GRAYSCALE)
image = image.astype(numpy.uint8)
image_height, image_width = image.shape[:2]

# Create the distortion object
real_distortion = ZernikeDistortion(parameters=parameters)
if DISPLAY_INTERMEDIATE:
    print(f"real distortion object:\n{real_distortion}\n")

# Rescale the distortion model to the image size
radius = numpy.sqrt((image_width / 2) ** 2 + (image_height / 2) ** 2)
center = (image_width / 2, image_height / 2)
real_distortion.radius = radius
real_distortion.center = center

# Distort the image
distorted_image = cv2_distort_image(src=image, K=None, distortion=real_distortion, method="undistort", interpolation="cubic")
distorted_image = distorted_image.astype(numpy.uint8)

# Save the distorted image
cv2.imwrite(distorted_image_file, distorted_image)

# Compute the real displacement field
pixel_points = numpy.indices((image_height, image_width), dtype=numpy.float64) # shape (2, H, W)
pixel_points = pixel_points.reshape(2, -1).T  # shape (H*W, 2) WARNING: [H, W -> Y, X]
pixel_points = pixel_points[:, [1, 0]]  # Swap to [X, Y] format

real_distorted_points, _, _ = real_distortion._transform(pixel_points, dx=False, dp=False)
real_displacement_field = real_distorted_points - pixel_points












#%%
# ===============================================================================
# ============== DISPLAY THE IMAGES AND REAL DISPLACEMENT FIELD =================
# ===============================================================================

if DISPLAY_INTERMEDIATE:
        
    # Display the original and distorted images
    fig_create_images = plt.figure(figsize=(10, 5))
    ax1 = fig_create_images.add_subplot(1, 2, 1)
    ax1.imshow(image, cmap='gray')
    ax1.set_title("Original Image")
    ax1.axis('off')
    ax2 = fig_create_images.add_subplot(1, 2, 2)
    ax2.imshow(distorted_image, cmap='gray')
    ax2.set_title("Distorted Image")
    ax2.axis('off')
    plt.tight_layout()

    # Display the real displacement field
    jump = 20
    magn_vmin = 0.0
    magn_vmax = 20.0
    axis_vmin = -10.0
    axis_vmax = 10.0
    fig_real_displacement_field = plt.figure(figsize=(15, 5))
    ax1 = fig_real_displacement_field.add_subplot(1, 3, 1)
    ax1.scatter(pixel_points[::jump, 0], pixel_points[::jump, 1], c=numpy.linalg.norm(real_displacement_field[::jump], axis=1), cmap='seismic', s=20, edgecolor='none', vmin=magn_vmin, vmax=magn_vmax)
    ax1_colorbar = plt.colorbar(ax1.collections[0], ax=ax1, orientation='vertical')
    ax1.set_title("Real Displacement Field (Magnitude)")
    ax1.set_aspect('equal')
    ax2 = fig_real_displacement_field.add_subplot(1, 3, 2)
    ax2.scatter(pixel_points[::jump, 0], pixel_points[::jump, 1], c=real_displacement_field[::jump, 0], cmap='seismic', s=20, edgecolor='none', vmin=axis_vmin, vmax=axis_vmax)
    ax2_colorbar = plt.colorbar(ax2.collections[0], ax=ax2, orientation='vertical')
    ax2.set_title("Real Displacement Field (X)")
    ax2.set_aspect('equal')
    ax3 = fig_real_displacement_field.add_subplot(1, 3, 3)
    ax3.scatter(pixel_points[::jump, 0], pixel_points[::jump, 1], c=real_displacement_field[::jump, 1], cmap='seismic', s=20, edgecolor='none', vmin=axis_vmin, vmax=axis_vmax)
    ax3_colorbar = plt.colorbar(ax3.collections[0], ax=ax3, orientation='vertical')
    ax3.set_title("Real Displacement Field (Y)")
    ax3.set_aspect('equal')
    plt.tight_layout()










# %%
# ===============================================================================
# ============== Measure the optical flow field =================================
# ===============================================================================
dis_optical_flow = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)

_supset = 50 # Size of the patch used for the optical flow estimation
_step = 10 # Step size for the patch sliding window

dis_optical_flow.setFinestScale(0)
dis_optical_flow.setVariationalRefinementAlpha(1)
dis_optical_flow.setVariationalRefinementDelta(1)
dis_optical_flow.setVariationalRefinementGamma(1)
dis_optical_flow.setVariationalRefinementEpsilon(0.02)
dis_optical_flow.setVariationalRefinementIterations(10)
dis_optical_flow.setUseMeanNormalization(True)
dis_optical_flow.setUseMeanNormalization(True)
dis_optical_flow.setGradientDescentIterations(500)
dis_optical_flow.setUseSpatialPropagation(True)
dis_optical_flow.setPatchSize(_supset)
dis_optical_flow.setPatchStride(_step)









# %%
# ===============================================================================
# ===== OPTIMIZE THE DISTORTION PARAMETERS (With Noise on displacment) ==========
# ===============================================================================

Nzer = 7 # Maximum order of Zernike polynomials to use
real_distortion_cropped_extended = copy.deepcopy(real_distortion)
real_distortion_cropped_extended.Nzer = Nzer

optimized_real_distortion = ZernikeDistortion(Nzer=Nzer)
optimized_real_distortion.radius = real_distortion.radius
optimized_real_distortion.center = real_distortion.center

# Select the disk for points used in computation.
radius_optimization = image_width / 2
center_optimization = (image_width / 2, image_height / 2)
mask_optimisation = numpy.sqrt((pixel_points[:,0] - center_optimization[0])**2 + (pixel_points[:,1] - center_optimization[1])**2) <= radius_optimization

# Parameters for optimization
mas_iter = 1
eps = 1e-8
reg_factor = 0.0
precond_jacobi = True 
cond_cutoff = 1e8

# Optimize the distortion parameters using the pixels points and the associated distorted points
optimized_real_parameters = optimized_real_distortion.optimize_parameters(
    input_points=pixel_points[mask_optimisation, :],
    output_points=real_distorted_points[mask_optimisation, :],
    guess=None, # Use the distortion parameters as initial guess (here zero because the object is initialized with default parameters)
    max_iter=mas_iter,
    eps=eps,
    reg_factor=reg_factor, # Regularization factor (0.0 means no regularization)
    precond_jacobi=precond_jacobi, # Whether to use the Jacobi preconditioner
    cond_cutoff=cond_cutoff, # Cutoff for the condition number of the Jacobian matrix
    verbose=DISPLAY_INTERMEDIATE,
    _verbose_eigen=DISPLAY_INTERMEDIATE, # Whether to display the eigenvalues and eigenvectors of the Jacobian matrix
)

# Compute the optimized displacement field
optimized_real_distortion.parameters = optimized_real_parameters
optimized_real_distorted_points, _, _ = optimized_real_distortion._transform(pixel_points, dx=False, dp=False)
optimized_real_displacement_field = optimized_real_distorted_points - pixel_points

# Loop on the noise in the displacement field
noise_std = numpy.linspace(0.0, 50, 20)
Nmean = 20
distortion_list = []
std_distortion_list = []

for index, noise in enumerate(noise_std):

    print(f"Optimizing distortion parameters for noise amplitude {noise:.2f} pixels ({index + 1}/{len(noise_std)})...")
    parameters_list = []

    # Create a new distortion object for optimization
    optimize_distortion = ZernikeDistortion(Nzer=Nzer)
    optimize_distortion.radius = real_distortion.radius
    optimize_distortion.center = real_distortion.center

    # Create a new distortion for std
    std_optimize_distortion = ZernikeDistortion(Nzer=Nzer)
    std_optimize_distortion.radius = real_distortion.radius
    std_optimize_distortion.center = real_distortion.center

    for iter in range(Nmean):

        # Add noise to the distorted image (Gaussian + Poisson + Salt & Pepper)
        # [Clean image]
        #      ↓
        # + Gaussian noise (sensor/electronic noise)
        #      ↓
        # + Poisson noise (photon/statistical noise)
        #      ↓
        # + Salt & pepper noise (transmission/memory defects)
        #      ↓
        # [Final noisy image]
        dyn_range = 255 # ! Change this value if the image is not in [0, 255] range
        noise_gaussian_var = (noise / dyn_range) ** 2  # Variance for Gaussian noise relative to the dynamic range
        noise_poisson = False  # Whether to add Poisson noise
        noise_salt_pepper_amount = 0.0 # Amount of salt & pepper noise

        noisy_distorted_image = skimage.util.random_noise(distorted_image, mode="gaussian", mean=0.0, var=noise_gaussian_var)
        if noise_poisson:
            noisy_distorted_image = skimage.util.random_noise(noisy_distorted_image, mode="poisson")
        if noise_salt_pepper_amount > 0.0:
            noisy_distorted_image = skimage.util.random_noise(noisy_distorted_image, mode="s&p", amount=noise_salt_pepper_amount)
        noisy_distorted_image = cv2.normalize(noisy_distorted_image, None, 0, 255, cv2.NORM_MINMAX).astype(numpy.uint8)

        if iter == 0:
            noise_file = os.path.join(os.path.dirname(__file__), "files", f"noisy_distorted_speckle_{index:02d}.png")
            cv2.imwrite(noise_file, noisy_distorted_image)

        # Normalize the images to [0, 255] range for optical flow calculation
        image_0 = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)
        image_1 = cv2.normalize(noisy_distorted_image, None, 0, 255, cv2.NORM_MINMAX)

        # Compute the optical flow
        flow = dis_optical_flow.calc(image_0, image_1, None) # shape (H, W, 2)

        # Reshape the flow to a 2D array of displacements
        flow_displacement_field = flow.astype(numpy.float64).reshape(-1, 2)  # shape (H*W, 2)
        flow_distorted_points = pixel_points + flow_displacement_field  # shape (H*W, 2)

        # Create a new distortion object for optimization
        optimize_iter_distortion = ZernikeDistortion(Nzer=Nzer)
        optimize_iter_distortion.radius = real_distortion.radius
        optimize_iter_distortion.center = real_distortion.center

        optimized_parameters = optimize_iter_distortion.optimize_parameters(
            input_points=pixel_points[mask_optimisation, :],
            output_points=flow_distorted_points[mask_optimisation, :],
            guess=None, # Use the distortion parameters as initial guess (here zero because the object is initialized with default parameters)
            max_iter=mas_iter,
            eps=eps,
            reg_factor=reg_factor, # Regularization factor (0.0 means no regularization)
            precond_jacobi=precond_jacobi, # Whether to use the Jacobi preconditioner
            cond_cutoff=cond_cutoff, # Cutoff for the condition number of the Jacobian matrix
            verbose=False,
            _verbose_eigen=False # Whether to display the eigenvalues and eigenvectors of the Jacobian matrix
        )

        # Append the optimized parameters to the list
        parameters_list.append(optimized_parameters)

    # Compute the mean and standard deviation of the optimized parameters
    optimized_parameters = numpy.mean(parameters_list, axis=0)
    std_parameters = numpy.std(parameters_list, axis=0)

    # Set the optimized parameters to the distortion object
    optimize_distortion.parameters = optimized_parameters
    std_optimize_distortion.parameters = std_parameters

    # Save the optimized distortion to the list
    distortion_list.append(optimize_distortion)
    std_distortion_list.append(std_optimize_distortion)







# %%
# ===============================================================================
# ============== DISPLAY THE DISPLACEMENT FOR A GIVEN NOISE  ====================
# ===============================================================================

if DISPLAY_INTERMEDIATE:
   
    noise_index = 5  # Index of the noise to display
    print(f"Displaying the distortion parameters for noise index {noise_index} : {noise_std[noise_index]} gaussian STD of gray level")

    noise_distortion = distortion_list[noise_index]
    std_noise_distortion = std_distortion_list[noise_index]
    noise_distorted_points, _, _ = noise_distortion._transform(pixel_points, dx=False, dp=False)
    noise_displacement_field = noise_distorted_points - pixel_points

    # Display the displacement fields
    jump = 20
    magn_vmin = 0.0
    magn_vmax = 20.0
    axis_vmin = -10.0
    axis_vmax = 10.0
    abs_error_vmin = 0.0
    abs_error_vmax = 1.0
    error_vmin = -1.0
    error_vmax = 1.0
    fig_optimized_displacement_fields = plt.figure(figsize=(15, 15))
    ax1 = fig_optimized_displacement_fields.add_subplot(3, 3, 1)
    ax1.scatter(pixel_points[::jump, 0], pixel_points[::jump, 1], c=numpy.linalg.norm(real_displacement_field[::jump], axis=1), cmap='seismic', s=20, edgecolor='none', vmin=magn_vmin, vmax=magn_vmax)
    ax1_colorbar = plt.colorbar(ax1.collections[0], ax=ax1, orientation='vertical')
    ax1.set_title("Real Displacement Field (Magnitude)")
    ax1.set_aspect('equal')
    ax2 = fig_optimized_displacement_fields.add_subplot(3, 3, 2)
    ax2.scatter(pixel_points[::jump, 0], pixel_points[::jump, 1], c=real_displacement_field[::jump, 0], cmap='seismic', s=20, edgecolor='none', vmin=axis_vmin, vmax=axis_vmax)
    ax2_colorbar = plt.colorbar(ax2.collections[0], ax=ax2, orientation='vertical')
    ax2.set_title("Real Displacement Field (X)")
    ax2.set_aspect('equal')
    ax3 = fig_optimized_displacement_fields.add_subplot(3, 3, 3)
    ax3.scatter(pixel_points[::jump, 0], pixel_points[::jump, 1], c=real_displacement_field[::jump, 1], cmap='seismic', s=20, edgecolor='none', vmin=axis_vmin, vmax=axis_vmax)
    ax3_colorbar = plt.colorbar(ax3.collections[0], ax=ax3, orientation='vertical')
    ax3.set_title("Real Displacement Field (Y)")
    ax3.set_aspect('equal')
    ax4 = fig_optimized_displacement_fields.add_subplot(3, 3, 4)
    ax4.scatter(pixel_points[::jump, 0], pixel_points[::jump, 1], c=numpy.linalg.norm(optimized_real_displacement_field[::jump], axis=1), cmap='seismic', s=20, edgecolor='none', vmin=magn_vmin, vmax=magn_vmax)
    ax4_colorbar = plt.colorbar(ax4.collections[0], ax=ax4, orientation='vertical')
    ax4.set_title("Optimized Real Displacement Field (Magnitude)")
    ax4.set_aspect('equal')
    ax5 = fig_optimized_displacement_fields.add_subplot(3, 3, 5)
    ax5.scatter(pixel_points[::jump, 0], pixel_points[::jump, 1], c=optimized_real_displacement_field[::jump, 0], cmap='seismic', s=20, edgecolor='none', vmin=axis_vmin, vmax=axis_vmax)
    ax5_colorbar = plt.colorbar(ax5.collections[0], ax=ax5, orientation='vertical')
    ax5.set_title("Optimized Real Displacement Field (X)")
    ax5.set_aspect('equal')
    ax6 = fig_optimized_displacement_fields.add_subplot(3, 3, 6)
    ax6.scatter(pixel_points[::jump, 0], pixel_points[::jump, 1], c=optimized_real_displacement_field[::jump, 1], cmap='seismic', s=20, edgecolor='none', vmin=axis_vmin, vmax=axis_vmax)
    ax6_colorbar = plt.colorbar(ax6.collections[0], ax=ax6, orientation='vertical')
    ax6.set_title("Optimized Real Displacement Field (Y)")
    ax6.set_aspect('equal')
    ax7 = fig_optimized_displacement_fields.add_subplot(3, 3, 7)
    ax7.scatter(pixel_points[::jump, 0], pixel_points[::jump, 1], c=numpy.linalg.norm(real_displacement_field[::jump] - optimized_real_displacement_field[::jump], axis=1), cmap='seismic', s=20, edgecolor='none', vmin=abs_error_vmin, vmax=abs_error_vmax)
    ax7_colorbar = plt.colorbar(ax7.collections[0], ax=ax7, orientation='vertical')
    ax7.set_title("Difference (Magnitude)")
    ax7.set_aspect('equal')
    ax8 = fig_optimized_displacement_fields.add_subplot(3, 3, 8)
    ax8.scatter(pixel_points[::jump, 0], pixel_points[::jump, 1], c=(real_displacement_field[::jump, 0] - optimized_real_displacement_field[::jump, 0]), cmap='seismic', s=20, edgecolor='none', vmin=error_vmin, vmax=error_vmax)
    ax8_colorbar = plt.colorbar(ax8.collections[0], ax=ax8, orientation='vertical')
    ax8.set_title("Difference (X)")
    ax8.set_aspect('equal')
    ax9 = fig_optimized_displacement_fields.add_subplot(3, 3, 9)
    ax9.scatter(pixel_points[::jump, 0], pixel_points[::jump, 1], c=(real_displacement_field[::jump, 1] - optimized_real_displacement_field[::jump, 1]), cmap='seismic', s=20, edgecolor='none', vmin=error_vmin, vmax=error_vmax)
    ax9_colorbar = plt.colorbar(ax9.collections[0], ax=ax9, orientation='vertical')
    ax9.set_title("Difference (Y)")
    ax9.set_aspect('equal')
    plt.tight_layout()


    # Display the flow displacement fields
    fig_noise_displacement_fields = plt.figure(figsize=(15, 15))
    ax1 = fig_noise_displacement_fields.add_subplot(3, 3, 1)
    ax1.scatter(pixel_points[::jump, 0], pixel_points[::jump, 1], c=numpy.linalg.norm(real_displacement_field[::jump], axis=1), cmap='seismic', s=20, edgecolor='none', vmin=magn_vmin, vmax=magn_vmax)
    ax1_colorbar = plt.colorbar(ax1.collections[0], ax=ax1, orientation='vertical')
    ax1.set_title("Real Displacement Field (Magnitude)")
    ax1.set_aspect('equal')
    ax2 = fig_noise_displacement_fields.add_subplot(3, 3, 2)
    ax2.scatter(pixel_points[::jump, 0], pixel_points[::jump, 1], c=real_displacement_field[::jump, 0], cmap='seismic', s=20, edgecolor='none', vmin=axis_vmin, vmax=axis_vmax)
    ax2_colorbar = plt.colorbar(ax2.collections[0], ax=ax2, orientation='vertical')
    ax2.set_title("Real Displacement Field (X)")
    ax2.set_aspect('equal')
    ax3 = fig_noise_displacement_fields.add_subplot(3, 3, 3)
    ax3.scatter(pixel_points[::jump, 0], pixel_points[::jump, 1], c=real_displacement_field[::jump, 1], cmap='seismic', s=20, edgecolor='none', vmin=axis_vmin, vmax=axis_vmax)
    ax3_colorbar = plt.colorbar(ax3.collections[0], ax=ax3, orientation='vertical')
    ax3.set_title("Real Displacement Field (Y)")
    ax3.set_aspect('equal')
    ax4 = fig_noise_displacement_fields.add_subplot(3, 3, 4)
    ax4.scatter(pixel_points[::jump, 0], pixel_points[::jump, 1], c=numpy.linalg.norm(noise_displacement_field[::jump], axis=1), cmap='seismic', s=20, edgecolor='none', vmin=magn_vmin, vmax=magn_vmax)
    ax4_colorbar = plt.colorbar(ax4.collections[0], ax=ax4, orientation='vertical')
    ax4.set_title("Optimized Noise Displacement Field (Magnitude)")
    ax4.set_aspect('equal')
    ax5 = fig_noise_displacement_fields.add_subplot(3, 3, 5)
    ax5.scatter(pixel_points[::jump, 0], pixel_points[::jump, 1], c=noise_displacement_field[::jump, 0], cmap='seismic', s=20, edgecolor='none', vmin=axis_vmin, vmax=axis_vmax)
    ax5_colorbar = plt.colorbar(ax5.collections[0], ax  =ax5, orientation='vertical')
    ax5.set_title("Optimized Noise Displacement Field (X)")
    ax5.set_aspect('equal')
    ax6 = fig_noise_displacement_fields.add_subplot(3, 3, 6)
    ax6.scatter(pixel_points[::jump, 0], pixel_points[::jump, 1], c=noise_displacement_field[::jump, 1], cmap='seismic', s=20, edgecolor='none', vmin=axis_vmin, vmax=axis_vmax)
    ax6_colorbar = plt.colorbar(ax6.collections[0], ax=ax6, orientation='vertical')
    ax6.set_title("Optimized Noise Displacement Field (Y)")
    ax6.set_aspect('equal')
    ax7 = fig_noise_displacement_fields.add_subplot(3, 3, 7)
    ax7.scatter(pixel_points[::jump, 0], pixel_points[::jump, 1], c=numpy.linalg.norm(real_displacement_field[::jump] - noise_displacement_field[::jump], axis=1), cmap='seismic', s=20, edgecolor='none', vmin=abs_error_vmin, vmax=abs_error_vmax)
    ax7_colorbar = plt.colorbar(ax7.collections[0], ax=ax7, orientation='vertical')
    ax7.set_title("Difference (Magnitude)")
    ax7.set_aspect('equal')
    ax8 = fig_noise_displacement_fields.add_subplot(3, 3, 8)
    ax8.scatter(pixel_points[::jump, 0], pixel_points[::jump, 1], c=(real_displacement_field[::jump, 0] - noise_displacement_field[::jump, 0]), cmap='seismic', s=20, edgecolor='none', vmin=error_vmin, vmax=error_vmax)
    ax8_colorbar = plt.colorbar(ax8.collections[0], ax=ax8, orientation='vertical')
    ax8.set_title("Difference (X)")
    ax8.set_aspect('equal')
    ax9 = fig_noise_displacement_fields.add_subplot(3, 3, 9)
    ax9.scatter(pixel_points[::jump, 0], pixel_points[::jump, 1], c=(real_displacement_field[::jump, 1] - noise_displacement_field[::jump, 1]), cmap='seismic', s=20, edgecolor='none', vmin=error_vmin, vmax=error_vmax)
    ax9_colorbar = plt.colorbar(ax9.collections[0], ax=ax9, orientation='vertical')
    ax9.set_title("Difference (Y)")
    ax9.set_aspect('equal')
    plt.tight_layout()


    # Display the optimized distortion parameters
    labels_x = [f"Cx({n}, {m})" for n in range(Nzer + 1) for m in range(-n, n + 1) if (n + m) % 2 == 0]
    labels_y = [f"Cy({n}, {m})" for n in range(Nzer + 1) for m in range(-n, n + 1) if (n + m) % 2 == 0]
    x = numpy.arange(len(labels_x))  # Position of the bars on the x-axis
    x_width = 0.20  # Width of the bars

    fig_optimzed_parameters = plt.figure(figsize=(10,20))
    ax1 = fig_optimzed_parameters.add_subplot(2, 1, 1)
    ax1.bar(x - x_width, real_distortion_cropped_extended.parameters_x, width=x_width, label='Real Distortion (Cx)', color='blue')
    ax1.bar(x, optimized_real_distortion.parameters_x, width=x_width, label='Optimized Real Distortion (Cx)', color='orange')
    ax1.bar(x + x_width, noise_distortion.parameters_x, width=x_width, label='Optimized Distortion (Cx)', color='green')
    ax1.errorbar(x + x_width, noise_distortion.parameters_x, yerr=std_noise_distortion.parameters_x, fmt='none', ecolor='black', capsize=5, label='Std Dev Noise Distortion (Cx)')
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels_x, rotation=90)
    ax1.set_title("Optimized Distortion Parameters (Cx) for Noise Amplitude = {:.2f} pixels".format(noise_std[noise_index]))
    ax1.set_ylabel("Parameter Value")
    ax1.legend()
    ax2 = fig_optimzed_parameters.add_subplot(2, 1, 2)
    ax2.bar(x - x_width, real_distortion_cropped_extended.parameters_y, width=x_width, label='Real Distortion (Cy)', color='blue')
    ax2.bar(x, optimized_real_distortion.parameters_y, width=x_width, label='Optimized Real Distortion (Cy)', color='orange')
    ax2.bar(x + x_width, noise_distortion.parameters_y, width=x_width, label='Optimized Distortion (Cy)', color='green')
    ax2.errorbar(x + x_width, noise_distortion.parameters_y, yerr=std_noise_distortion.parameters_y, fmt='none', ecolor='black', capsize=5, label='Std Dev Noise Distortion (Cy)')
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels_y, rotation=90)
    ax2.set_title("Optimized Distortion Parameters (Cy) for Noise Amplitude = {:.2f} pixels".format(noise_std[noise_index]))
    ax2.set_ylabel("Parameter Value")
    ax2.legend()
    plt.tight_layout()






# %%
# ===============================================================================
# ==== DISPLAY THE OPTIMIZED DISTORTION PARAMETERS For various noise ============
# ===============================================================================


# Display the optimized distortion parameters
labels_x = [f"Cx({n}, {m})" for n in range(Nzer + 1) for m in range(-n, n + 1) if (n + m) % 2 == 0]
labels_y = [f"Cy({n}, {m})" for n in range(Nzer + 1) for m in range(-n, n + 1) if (n + m) % 2 == 0]
x = numpy.arange(len(labels_x))  # Position of the bars on the x-axis
x_width = 0.50  # Width of the bars
colormap = plt.get_cmap('viridis')

fig_optimized_parameters = plt.figure(figsize=(30,30))
fig_optimized_parameters.suptitle("Distortion Parameters for various gaussian STD of gray level image noise \n I = I + gauss(0, noise)", fontsize=16, fontweight='bold')

ax1 = fig_optimized_parameters.add_subplot(2, 2, 1)

for i, (noise_distortion, std_noise_distortion) in enumerate(zip(distortion_list, std_distortion_list)):
    ax1.bar(x + i * x_width / len(distortion_list), noise_distortion.parameters_x, width=x_width / len(distortion_list), label=f'(Cx) - Noise {noise_std[i]:.2f} STD', color=colormap(i / len(distortion_list)))

ax1.set_xticks(x)
ax1.set_xticklabels(labels_x, rotation=90)
ax1.set_title("Optimized Distortion Parameters (Cx) (mean values)")
ax1.set_ylabel("Parameter Value")
ax1.legend()

ax2 = fig_optimized_parameters.add_subplot(2, 2, 2)

for i, (noise_distortion, std_noise_distortion) in enumerate(zip(distortion_list, std_distortion_list)):
    ax2.bar(x + i * x_width / len(distortion_list), noise_distortion.parameters_y, width=x_width / len(distortion_list), label=f'(Cy) - Noise {noise_std[i]:.2f} STD', color=colormap(i / len(distortion_list)))
    
ax2.set_xticks(x)
ax2.set_xticklabels(labels_y, rotation=90)
ax2.set_title("Optimized Distortion Parameters (Cy) (mean values)")
ax2.set_ylabel("Parameter Value")
ax2.legend()

ax3 = fig_optimized_parameters.add_subplot(2, 2, 3)

for i, (noise_distortion, std_noise_distortion) in enumerate(zip(distortion_list, std_distortion_list)):
    ax3.bar(x + i * x_width / len(distortion_list), noise_distortion.parameters_x, width=x_width / len(distortion_list), label=f'(Cx) - Noise {noise_std[i]:.2f} STD', color=colormap(i / len(distortion_list)))
    ax3.errorbar(x + i * x_width / len(distortion_list), noise_distortion.parameters_x, yerr=std_noise_distortion.parameters_x, fmt='none', ecolor='black', capsize=5)

ax3.set_xticks(x)
ax3.set_xticklabels(labels_x, rotation=90)
ax3.set_title("Optimized Distortion Parameters (Cx)")
ax3.set_ylabel("Parameter Value")
ax3.legend()

ax4 = fig_optimized_parameters.add_subplot(2, 2, 4)

for i, (noise_distortion, std_noise_distortion) in enumerate(zip(distortion_list, std_distortion_list)):
    ax4.bar(x + i * x_width / len(distortion_list), noise_distortion.parameters_y, width=x_width / len(distortion_list), label=f'(Cy) - Noise {noise_std[i]:.2f} STD', color=colormap(i / len(distortion_list)))
    ax4.errorbar(x + i * x_width / len(distortion_list), noise_distortion.parameters_y, yerr=std_noise_distortion.parameters_y, fmt='none', ecolor='black', capsize=5)
    
ax4.set_xticks(x)
ax4.set_xticklabels(labels_y, rotation=90)
ax4.set_title("Optimized Distortion Parameters (Cy)")
ax4.set_ylabel("Parameter Value")
ax4.legend()

plt.tight_layout()
fig_optimized_parameters.subplots_adjust(top=0.95)  # Adjust the top to make space for the suptitle
plt.show()


# %%
# ===============================================================================
# ============== SAVE THE FIGURES ===============================================
# ===============================================================================

figsave = os.path.join(os.path.dirname(__file__), "result", "optimized_distortion_parameters_higher.png")
if not os.path.exists(os.path.dirname(figsave)):
    os.makedirs(os.path.dirname(figsave))

fig_optimized_parameters.savefig(figsave, bbox_inches='tight', dpi=300)



# %%
