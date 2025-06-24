import numpy as np 

# Create default points 
points = np.random.rand(5, 6, 3) # (..., 2 + <homogeneous>) with ... = (5, 6)
print("points.shape", points.shape) # (5, 6, 3)

# Transpose points to move the last axis to the front
points_transposed = np.moveaxis(points, -1, 0) # (..., 2 + <homogeneous>) -> (2 + <homogeneous>, ...)
print("points_transposed.shape", points_transposed.shape) # (3, 5, 6)

points_flat = points_transposed.reshape(points_transposed.shape[0], -1) # (2 + <homogeneous>, ...) -> (2 + <homogeneous>, Npoints)
print("points_flat.shape", points_flat.shape) # (3, 30)

jacobian_flat = np.random.rand(2, 30, 7) # (2 + <homogeneous>, Npoints, 2 + Nparams)
print("jacobian_flat.shape", jacobian_flat.shape) # (2, 30, 7)

jacobian_flat = np.concatenate([jacobian_flat, np.repeat(points_flat[2:, ..., None], 7, axis=-1)], axis=0) # (2, Npoints, 2 + Nparams) -> (2 + <homogeneous>, Npoints, 2 + Nparams)
print("jacobian_flat.shape", jacobian_flat.shape) # (3, 30, 7)

jacobian = jacobian_flat.reshape((*points_transposed.shape, -1)) # (2 + <homogeneous>, Npoints, 2 + Nparams) -> (2 + <homogeneous>, ..., 2 + Nparams)
print("jacobian.shape", jacobian.shape) # (3, 5, 6, 7)

jacobian_transposed = np.moveaxis(jacobian, 0, -2) # (2 + <homogeneous>, ..., 2 + Nparams) -> (..., 2 + <homogeneous>, 2 + Nparams)
print("jacobian_transposed.shape", jacobian_transposed.shape) # (5, 6, 3, 7)