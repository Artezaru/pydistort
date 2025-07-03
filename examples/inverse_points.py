from pydistort import ZernikeDistortion
import numpy
import cv2
import os
import matplotlib.pyplot as plt


def ZERNIKE_DISTORTION():
    """Create a ZernikeDistortion object with specified number of Zernike coefficients."""
    distortion = ZernikeDistortion(Nzer=7)

    # Set Zernike coefficients
    distortion.set_Cx(0, 0, 0.017083945091492785)
    distortion.set_Cy(0, 0, -0.1093719257958107)
    distortion.set_Cx(1, 1, 0.04280641095874525)
    distortion.set_Cx(1, -1, -0.11948575638043393)
    distortion.set_Cy(1, 1, 0.0908833886027441)
    distortion.set_Cy(1, -1, 0.28585912150232207)
    distortion.set_Cx(2, 0, -0.010212748711363793)
    distortion.set_Cy(2, 0, -0.11540175375301409)
    distortion.set_Cx(2, 2, -0.00782950115774214)
    distortion.set_Cx(2, -2, -0.0020199464928398678)
    distortion.set_Cy(2, 2, 0.23398822546004996)
    distortion.set_Cy(2, -2, 0.008727018408134835)
    distortion.set_Cx(3, 1, 0.11774670344474367)
    distortion.set_Cx(3, -1, -0.03842086254300457)
    distortion.set_Cy(3, 1, 0.015958056702810412)
    distortion.set_Cy(3, -1, 0.4053713119884255)
    distortion.set_Cx(3, 3, -0.06941369934820552)
    distortion.set_Cx(3, -3, 0.06858990952409365)
    distortion.set_Cy(3, 3, -0.058872634305352195)
    distortion.set_Cy(3, -3, -0.27273893460948323)
    distortion.set_Cx(4, 0, -0.0008355538427007839)
    distortion.set_Cy(4, 0, -0.07902677499990182)
    distortion.set_Cx(4, 2, -0.002596009621418076)
    distortion.set_Cx(4, -2, -0.0004671581111743396)
    distortion.set_Cy(4, 2, 0.1622500117071097)
    distortion.set_Cy(4, -2, 0.009242023070156922)
    distortion.set_Cx(4, 4, -0.0016053903604748264)
    distortion.set_Cx(4, -4, 0.003055958197544206)
    distortion.set_Cy(4, 4, -0.16733400168088858)
    distortion.set_Cy(4, -4, -0.016179979676594455)
    distortion.set_Cx(5, 1, 0.012583260318218136)
    distortion.set_Cx(5, -1, -0.015945506008503228)
    distortion.set_Cy(5, 1, 0.005112296621836569)
    distortion.set_Cy(5, -1, 0.1653781339398673)
    distortion.set_Cx(5, 3, -0.03108848626999168)
    distortion.set_Cx(5, -3, 0.034148224370183465)
    distortion.set_Cy(5, 3, -0.029328096123159834)
    distortion.set_Cy(5, -3, -0.13984684802950417)
    distortion.set_Cx(5, 5, 0.08936881806050903)
    distortion.set_Cx(5, -5, -0.06411318032885825)
    distortion.set_Cy(5, 5, 0.0725410316989168)
    distortion.set_Cy(5, -5, 0.10346426469415285)
    distortion.set_Cx(6, 0, 0.0002924901298748926)
    distortion.set_Cy(6, 0, -0.02322855580912703)
    distortion.set_Cx(6, 2, -0.00018960015221684552)
    distortion.set_Cx(6, -2, -0.00037298295468960515)
    distortion.set_Cy(6, 2, 0.049363725455249606)
    distortion.set_Cy(6, -2, 0.003670537559413702)
    distortion.set_Cx(6, 4, 0.0006314053384830566)
    distortion.set_Cx(6, -4, 0.001597676829797511)
    distortion.set_Cy(6, 4, -0.04884846328688792)
    distortion.set_Cy(6, -4, -0.005835730813113942)
    distortion.set_Cx(6, 6, 0.001941729950393342)
    distortion.set_Cx(6, -6, -0.001298495448916161)
    distortion.set_Cy(6, 6, 0.05154502761557224)
    distortion.set_Cy(6, -6, 0.005679929634243922)
    distortion.set_Cx(7, 1, 0.0003465309056524352)
    distortion.set_Cx(7, -1, 0.00029803470070771535)
    distortion.set_Cy(7, 1, -0.0005672611158840981)
    distortion.set_Cy(7, -1, 0.0477615940060408)
    distortion.set_Cx(7, 3, -0.0011746651422660356)
    distortion.set_Cx(7, -3, 0.00704874734072278)
    distortion.set_Cy(7, 3, -0.0025490979756813974)
    distortion.set_Cy(7, -3, -0.04020174877407482)
    distortion.set_Cx(7, 5, 0.015772380855886274)
    distortion.set_Cx(7, -5, -0.017702627357461236)
    distortion.set_Cy(7, 5, 0.012448080057913424)
    distortion.set_Cy(7, -5, 0.02824867723847335)
    distortion.set_Cx(7, 7, -0.048295980450963484)
    distortion.set_Cx(7, -7, 0.028571068337471497)
    distortion.set_Cy(7, 7, -0.03981484964981821)
    distortion.set_Cy(7, -7, -0.01254678379222499)

    return distortion

height, width = 2472, 3297

distortion = ZERNIKE_DISTORTION()

radius = numpy.sqrt((width / 2) ** 2 + (height / 2) ** 2)
center = (width / 2, height / 2)

distortion.radius = radius
distortion.center = center
distortion.parameters = distortion.parameters * 100

# Create the normalized points
pixel_points = numpy.indices((height, width), dtype=numpy.float64) # shape (2, H, W)
pixel_points = pixel_points.reshape(2, -1).T  # shape (H*W, 2) WARNING: [H, W -> Y, X]
pixel_points = pixel_points[:, [1, 0]]  # Swap to [X, Y] format
normalized_points = pixel_points.copy()  # shape (H*W, 2)
print(f"Normaliezed points : {normalized_points}")

distorted_points, _, _ = distortion._transform(normalized_points=normalized_points, dx=False, dp=False)

displacement = distorted_points - normalized_points

jump = 50

fig = plt.figure(figsize=(12, 6))
ax = fig.add_subplot(1, 2, 1)
ax.scatter(normalized_points[::jump, 0], normalized_points[::jump, 1], c=displacement[::jump, 0], s=10, cmap='viridis', label='Normalized Points')
colorbar = plt.colorbar(ax.collections[0], ax=ax, orientation='vertical')
ax = fig.add_subplot(1, 2, 2)
ax.scatter(distorted_points[::jump, 0], distorted_points[::jump, 1], c=displacement[::jump, 1], s=10, cmap='viridis', label='Distorted Points')
colorbar = plt.colorbar(ax.collections[0], ax=ax, orientation='vertical')

plt.show()

print(distorted_points - normalized_points)

undistorted_points, _, _ = distortion._inverse_transform(distorted_points=distorted_points, dx=False, dp=False, max_iter=10, eps=1e-8)

nan_mask = numpy.isnan(undistorted_points)
print("NaNs : ", numpy.sum(nan_mask))

assert numpy.allclose(normalized_points[~nan_mask], undistorted_points[~nan_mask], atol=1e-8)


# Display the original and distorted points
fig = plt.figure(figsize=(12, 6))
ax = fig.add_subplot(1, 2, 1)


ax.scatter(normalized_points[::jump, 0], normalized_points[::jump, 1], color='blue', s=1, label='Normalized Points')
ax.scatter(distorted_points[::jump, 0], distorted_points[::jump, 1], color='red', s=1, label='Distorted Points')

ax.set_title('Normalized and Distorted Points')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.legend()

ax = fig.add_subplot(1, 2, 2)
ax.scatter(normalized_points[::jump, 0], normalized_points[::jump, 1], color='blue', s=1, label='Normalized Points')
ax.scatter(distorted_points[::jump, 0], distorted_points[::jump, 1], color='red', s=1, label='Distorted Points')
ax.scatter(undistorted_points[::jump, 0], undistorted_points[::jump, 1], color='green', s=1, label='Undistorted Points')

ax.set_title('Normalized, Distorted, and Undistorted Points')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.legend()

plt.tight_layout()
plt.show()
