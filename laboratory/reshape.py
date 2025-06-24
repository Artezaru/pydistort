import numpy as np

# Exemple
points = np.random.rand(2, 5, 6)

# Aplatir
points_flat = points.reshape(2, -1)  # (2, 30)

# Restaurer
points_restored = points_flat.reshape(2, 5, 6)

# Vérification
print(np.allclose(points, points_restored))  # True ✅