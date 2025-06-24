from pydistort import Cv2Distortion, project_points
import numpy as np
import time 

def get_distortion(Nparams, mode):
    """Create a Cv2Distortion object with specified parameters."""
    distortion = Cv2Distortion(Nparams=Nparams)

    if mode == "strong_coefficients":
        distortion.k1 = 47.6469
        distortion.k2 = 605.372
        distortion.p1 = 0.01304
        distortion.p2 = -0.02737
        distortion.k3 = -1799.929
        if Nparams >= 8:
            distortion.k4 = 47.765
            distortion.k5 = 500.027
            distortion.k6 = 1810.745
        if Nparams >= 12:
            distortion.s1 = -0.0277
            distortion.s2 = 1.9759
            distortion.s3 = -0.0208
            distortion.s4 = 0.3596
        if Nparams == 14:
            distortion.taux = 2.0
            distortion.tauy = 5.0

    elif mode == "weak_coefficients":
        distortion.k1 = 1e-4
        distortion.k2 = 1e-5
        distortion.p1 = 1e-5
        distortion.p2 = 1e-5
        distortion.k3 = 1e-5
        if Nparams >= 8:
            distortion.k4 = 1e-5
            distortion.k5 = 1e-5
            distortion.k6 = 1e-5
        if Nparams >= 12:
            distortion.s1 = 1e-5
            distortion.s2 = 1e-5
            distortion.s3 = 1e-5
            distortion.s4 = 1e-5
        if Nparams == 14:
            distortion.taux = 1e-5
            distortion.tauy = 1e-5

    return distortion

distortion = get_distortion(Nparams=14, mode="strong_coefficients")

Npoints = 1000000

# Camera intrinsics
fx, fy = 1000.0, 950.0
cx, cy = 320.0, 240.0
K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])

# Rotation and translation
rvec = np.array([0.01, 0.02, 0.03])  # small rotation
tvec = np.array([0.1, -0.1, 0.2])    # small translation

# Test points
points = np.random.uniform(-1.0, 1.0, size=(Npoints, 2))  # shape (Npoints, 2)
points = np.concatenate((points, 5.0 * np.ones((Npoints, 1))), axis=1) # shape (Npoints, 3)

# Distortion (analytic)
start_time = time.time()
result = project_points(points, rvec=rvec, tvec=tvec, K=K, distortion=distortion, dx=True, dp=True)
elapsed_time = time.time() - start_time
print(f"Elapsed time for project_points: {elapsed_time:.6f} seconds")