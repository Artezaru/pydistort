import numpy as np
import cv2


object_points = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]], dtype=np.float32)

rvec = np.zeros((3, 1), dtype=np.float32)
tvec = np.zeros((3, 1), dtype=np.float32)
camera_matrix = np.eye(3, dtype=np.float32)
dist = np.zeros((8, 1), dtype=np.float32)

_, jacobian = cv2.projectPoints(
    object_points,
    rvec,
    tvec,
    camera_matrix,
)

print("Jacobian shape:", jacobian.shape)
print("Jacobian:", jacobian)