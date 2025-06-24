import numpy as np

print("--- 2D ---")
a = np.zeros((100, 2))
print(np.moveaxis(a, -1, 0).shape)  # OK (2, 100)

print("--- 3D ---")
b = np.zeros((5, 4, 2))
print(np.moveaxis(b, -1, 0).shape)  # OK (2, 5, 4)

print("--- 4D ---")
c = np.zeros((7, 5, 4, 2))
print(np.moveaxis(c, -1, 0).shape)  # OK (2, 7, 5, 4)
