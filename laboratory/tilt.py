import numpy as np

# Définition de la classe ou de la structure contenant les angles tau_x et tau_y
class TiltTransformation:
    def __init__(self, tau_x, tau_y):
        self.tau_x = tau_x
        self.tau_y = tau_y

    def get_Rmat(self):
        # Matrice de rotation Rmat
        Rmat = np.array([
            [np.cos(self.tau_y), np.sin(self.tau_x) * np.sin(self.tau_y), -np.cos(self.tau_x) * np.sin(self.tau_y)],
            [0, np.cos(self.tau_x), np.sin(self.tau_x)],
            [np.sin(self.tau_y), -np.sin(self.tau_x) * np.cos(self.tau_y), np.cos(self.tau_x) * np.cos(self.tau_y)]
        ], dtype=np.float64)
        return Rmat

    def get_Cormat(self, Rmat):
        # Matrice Cormat basée sur Rmat
        Cormat = np.array([
            [Rmat[2, 2], 0, -Rmat[0, 2]],
            [0, Rmat[2, 2], -Rmat[1, 2]],
            [0, 0, 1]
        ], dtype=np.float64)
        return Cormat

    def tilt_points(self, x, y):
        # Applique la transformation Cormat * Rmat * [x, y, 1]
        Rmat = self.get_Rmat()
        Cormat = self.get_Cormat(Rmat)

        # Vecteur de coordonnées homogènes [x, y, 1]
        point = np.array([x, y, 1], dtype=np.float64)

        # Effectuer la multiplication Cormat * Rmat * point
        result = Cormat @ (Rmat @ point)
        return result

# Exemple d'utilisation
tau_x = np.radians(90)  # Rotation autour de l'axe X (en radians)
tau_y = np.radians(45)  # Rotation autour de l'axe Y (en radians)

# Créer l'instance de la classe TiltTransformation
tilt_transform = TiltTransformation(tau_x, tau_y)

# Point à transformer
x, y = 1, 1  # Par exemple, le point (1, 1)

# Appliquer la transformation
transformed_point = tilt_transform.tilt_points(x, y)

print(f"Point original: ({x}, {y})")
print(f"Point transformé: {transformed_point}")
