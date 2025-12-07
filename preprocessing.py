import cv2
import numpy as np


def gaussian_smoothing(img):
    return cv2.GaussianBlur(img, (5, 5), 0)


def compute_gradient_magnitude(gray):
    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    mag = np.sqrt(gx ** 2 + gy ** 2)
    mag = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return mag


def abocnn_preprocessing(image):
    """
    Input: RGB uint8 image (H,W,3)
    Output: float32 image in [0,1] with enhanced edges.
    """
    image = image.astype("uint8")
    smoothed = gaussian_smoothing(image)
    gray = cv2.cvtColor(smoothed, cv2.COLOR_RGB2GRAY)
    grad_mag = compute_gradient_magnitude(gray)
    grad_rgb = cv2.cvtColor(grad_mag, cv2.COLOR_GRAY2RGB)
    enhanced = cv2.addWeighted(smoothed, 0.8, grad_rgb, 0.2, 0)
    return enhanced.astype(np.float32) / 255.0
