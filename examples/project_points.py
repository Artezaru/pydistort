from pydistort import project_points, ZernikeDistortion
import numpy as np

Npoints = 1_000_000
distortion = ZernikeDistortion(Nzer=7)
distortion.parameters = np.random.uniform(-1, 1, distortion.Nparams)

points = np.random.uniform(-1, 1, (Npoints, 3))

rvec = np.random.uniform(-np.pi, np.pi, 3)
tvec = np.random.uniform(-1, 1, 3)
K = np.array([[1000, 0, 320],
              [0, 1000, 240],
              [0, 0, 1]])

# Project points using the distortion model
points_projected = project_points(points, rvec, tvec, K, distortion)