import numpy as np
from scipy.spatial.transform import Rotation as R
import cv2

def test_rotvec_vs_cv2_rodrigues():
    # Exemple de vecteur rotation (angle = norme, axe = direction)
    rotvec = np.array([0.1, 0.2, 0.3])

    # --- SciPy: vecteur rotation → matrice rotation
    r_scipy = R.from_rotvec(rotvec)
    mat_scipy = r_scipy.as_matrix()

    # --- OpenCV: vecteur rotation → matrice rotation
    mat_cv2, _ = cv2.Rodrigues(rotvec)

    # Vérification que matrices sont proches
    print("Rotation matrix from SciPy:")
    print(mat_scipy)
    print("\nRotation matrix from OpenCV:")
    print(mat_cv2)
    print("\nDifference matrix:")
    print(mat_scipy - mat_cv2)

    assert np.allclose(mat_scipy, mat_cv2, atol=1e-8), "Matrices rotation diffèrent !"

    # --- Inverse: matrice rotation → vecteur rotation
    rotvec_scipy = R.from_matrix(mat_scipy).as_rotvec()
    rotvec_cv2, _ = cv2.Rodrigues(mat_cv2)

    print("\nRotation vector from SciPy (inverse):", rotvec_scipy)
    print("Rotation vector from OpenCV (inverse):", rotvec_cv2.ravel())
    print("Difference vecteurs rotation:", rotvec_scipy - rotvec_cv2.ravel())

    assert np.allclose(rotvec_scipy, rotvec_cv2.ravel(), atol=1e-8), "Rotation vectors differ!"

    print("\nTest passed: SciPy and OpenCV Rodrigues conversions are consistent.")

if __name__ == "__main__":
    test_rotvec_vs_cv2_rodrigues()
