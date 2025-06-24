from __future__ import print_function
import cv2 as cv
import numpy as np
import sys

def update_map(ind, map_x, map_y):
    if ind == 0:
        for i in range(map_x.shape[0]):
            for j in range(map_x.shape[1]):
                if j > map_x.shape[1]*0.25 and j < map_x.shape[1]*0.75 and i > map_x.shape[0]*0.25 and i < map_x.shape[0]*0.75:
                    map_x[i,j] = 2 * (j-map_x.shape[1]*0.25) + 0.5
                    map_y[i,j] = 2 * (i-map_y.shape[0]*0.25) + 0.5
                else:
                    map_x[i,j] = 0
                    map_y[i,j] = 0
    elif ind == 1:
        for i in range(map_x.shape[0]):
            map_x[i,:] = [x for x in range(map_x.shape[1])]
        for j in range(map_y.shape[1]):
            map_y[:,j] = [map_y.shape[0]-y for y in range(map_y.shape[0])]
    elif ind == 2:
        for i in range(map_x.shape[0]):
            map_x[i,:] = [map_x.shape[1]-x for x in range(map_x.shape[1])]
        for j in range(map_y.shape[1]):
            map_y[:,j] = [y for y in range(map_y.shape[0])]
    elif ind == 3:
        for i in range(map_x.shape[0]):
            map_x[i,:] = [map_x.shape[1]-x for x in range(map_x.shape[1])]
        for j in range(map_y.shape[1]):
            map_y[:,j] = [map_y.shape[0]-y for y in range(map_y.shape[0])]


src = cv.imread("ORI.jpg", cv.IMREAD_COLOR)

map_x = np.zeros((src.shape[0], src.shape[1]), dtype=np.float32)
map_y = np.zeros((src.shape[0], src.shape[1]), dtype=np.float32)

update_map(0, map_x, map_y)

# dst = cv.remap(src, map_x, map_y, cv.INTER_LINEAR)
# cv.imshow("Test", dst)
# c = cv.waitKey(0)
# sys.exit(0)


# Charger une image
image = cv.imread('ORI.jpg', cv.IMREAD_COLOR)  # Remplacez par le chemin de votre image
image = cv.resize(image, (10,10))  # Redimensionner pour un test rapide
H, W = image.shape[:2]  # Dimensions de l'image

# Créer les maps pour le remap
map_x = np.zeros((H, W), dtype=np.float32)
map_y = np.zeros((H, W), dtype=np.float32)
print("Dimensions de l'image :", H, "x", W)
print("Dimensions des maps :", map_x.shape, "x", map_y.shape)
# Remplir les maps avec des valeurs de remappage
for i in range(H):
    for j in range(W):
        map_x[i, j] = j
        map_y[i, j] = i

print(map_x)
print(map_y)

dst = cv.remap(image, map_x, map_y, cv.INTER_LINEAR)
assert np.allclose(dst, image), "Remapping failed, output does not match input image"
cv.imshow("Test", dst)
c = cv.waitKey(0)
sys.exit(0)
