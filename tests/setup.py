import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import json
import cv2
import numpy

from pydistort import Cv2Distortion, ZernikeDistortion
from pydistort import Cv2Intrinsic

def CSV():
    """Return is setup.json has 'csv' set to True."""
    filename = "setup.json"
    filename = os.path.join(os.path.dirname(__file__), filename)
    with open(filename, "r") as f:
        setup = json.load(f)
    return setup.get("csv", False)

def TIMER():
    """Return is setup.json has 'timer' set to True."""
    filename = "setup.json"
    filename = os.path.join(os.path.dirname(__file__), filename)
    with open(filename, "r") as f:
        setup = json.load(f)
    return setup.get("timer", False)

def DISPLAY():
    """Return is setup.json has 'display' set to True."""
    filename = "setup.json"
    filename = os.path.join(os.path.dirname(__file__), filename)
    with open(filename, "r") as f:
        setup = json.load(f)
    return setup.get("display", False)

def VERBOSE():
    """Return is setup.json has 'verbose' set to True."""
    filename = "setup.json"
    filename = os.path.join(os.path.dirname(__file__), filename)
    with open(filename, "r") as f:
        setup = json.load(f)
    return setup.get("verbose", False)

def VERBOSE_LEVEL():
    """Return the verbose level from setup.json."""
    filename = "setup.json"
    filename = os.path.join(os.path.dirname(__file__), filename)
    with open(filename, "r") as f:
        setup = json.load(f)
    return setup.get("verbose_level", 0)

def WARNINGS():
    """Return is setup.json has 'warnings' set to True."""
    filename = "setup.json"
    filename = os.path.join(os.path.dirname(__file__), filename)
    with open(filename, "r") as f:
        setup = json.load(f)
    return setup.get("warnings", False)




# ==========================================
# Some Data
# ==========================================

def ORI_IMAGE():
    """Return the path to the original image."""
    path = os.path.join(os.path.dirname(__file__), "ORI.jpg")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Original image not found at {path}")
    
    image = cv2.imread(path, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Failed to read the image at {path}")
    
    return image


def ORI_IMAGE_POINTS():
    """Return the points from the original image."""
    image = ORI_IMAGE()
    height, width = image.shape[:2]
    pixel_points = numpy.indices((height, width), dtype=numpy.float64) # shape (2, H, W)
    pixel_points = pixel_points.reshape(2, -1).T  # shape (H*W, 2) WARNING: [H, W -> Y, X]
    pixel_points = pixel_points[:, [1, 0]]  # Swap to [X, Y] format
    return pixel_points


def ORI_MATK():
    """Return the intrinsic matrix from the original image."""
    image = ORI_IMAGE()
    height, width = image.shape[:2]
    fx = 10000.0
    fy = 9500.0
    cx = width / 2.0
    cy = height / 2.0
    return numpy.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=numpy.float64)


def ORI_NORMALIZED_POINTS():
    """Return the normalized points from the original image."""
    image_points = ORI_IMAGE_POINTS()
    intrinsic = Cv2Intrinsic()
    intrinsic.intrinsic_matrix = ORI_MATK()
    normalized_points, _, _ = intrinsic._inverse_transform(image_points, dx=False, dp=False)
    return normalized_points


def CV2_DISTORTION(Nparams, mode):
    """Create a Cv2Distortion object with specified parameters."""
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

