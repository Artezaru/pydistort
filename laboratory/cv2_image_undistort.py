import numpy as np
import cv2
from pydistort import Cv2Distortion, undistort_image

def get_distortion(Nparams, mode):
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

def main():
    Nparams = 5
    mode = "strong_coefficients"

    distortion = get_distortion(Nparams, mode)

    # Camera intrinsics
    fx, fy = 1000.0, 950.0
    cx, cy = 320.0, 240.0
    K = np.array([[fx, 0, cx],
                  [0, fy, cy],
                  [0, 0, 1]])

    H, W = 480, 640
    image = np.zeros((H, W, 3), dtype=np.uint8)

    # Create a wave pattern image (red channel)
    for y in range(H):
        for x in range(W):
            val = int(127.5 * (1 + np.sin(2 * np.pi * (x / W + y / H))))
            image[y, x] = [val, 0, 0]

    # Undistort with pydistort
    result_pydistort = undistort_image(image, K=K, distortion=distortion)

    # Prepare distortion coefficients for OpenCV
    # OpenCV expects a 1D array with 5 or 8 params:
    # [k1, k2, p1, p2, k3, k4, k5, k6]
    # We'll fill missing values with zeros if Nparams < 8
    params = [distortion.k1, distortion.k2, distortion.p1, distortion.p2]
    if Nparams >= 5:
        params.append(distortion.k3)
    else:
        params.append(0)
    if Nparams >= 8:
        params.extend([distortion.k4, distortion.k5, distortion.k6])
    else:
        params.extend([0, 0, 0])
    params = np.array(params[:8], dtype=np.float64)

    # Undistort with OpenCV
    result_cv2 = cv2.undistort(image, K, params)

    # Show results side-by-side
    combined = np.hstack((result_pydistort, result_cv2))
    cv2.imshow("Pydistort (left) vs OpenCV (right)", combined)
    print("Press any key on the image window to exit")
    cv2.waitKey(0)
    import sys
    sys.exit(0)

if __name__ == "__main__":
    main()
