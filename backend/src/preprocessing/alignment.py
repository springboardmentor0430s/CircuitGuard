import cv2
import numpy as np

def _to_grayscale(img):
    """Convert image to grayscale if needed"""
    if len(img.shape) == 3:
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img.copy()

def align_images(test_img, template_img):
    """ORB feature-based alignment"""
    test_gray = _to_grayscale(test_img)
    template_gray = _to_grayscale(template_img)
    
    orb = cv2.ORB_create(5000)
    kp1, des1 = orb.detectAndCompute(test_gray, None)
    kp2, des2 = orb.detectAndCompute(template_gray, None)
    
    if des1 is None or des2 is None or not kp1 or not kp2:
        return None, None
    
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    
    if len(matches) < 4:
        return None, None
    
    matches = sorted(matches, key=lambda x: x.distance)
    points1 = np.array([kp1[m.queryIdx].pt for m in matches], dtype=np.float32)
    points2 = np.array([kp2[m.trainIdx].pt for m in matches], dtype=np.float32)
    
    H, mask = cv2.findHomography(points1, points2, cv2.RANSAC, 5.0)
    
    if H is not None:
        height, width = template_gray.shape
        aligned_img = cv2.warpPerspective(test_img, H, (width, height))
        return aligned_img, H
    
    return None, None

def simple_alignment(test_img, template_img):
    """Template matching alignment fallback"""
    test_gray = _to_grayscale(test_img)
    template_gray = _to_grayscale(template_img)
    
    h, w = template_gray.shape
    result = cv2.matchTemplate(test_gray, template_gray, cv2.TM_CCOEFF_NORMED)
    _, _, _, max_loc = cv2.minMaxLoc(result)
    
    top_left = max_loc
    bottom_right = (top_left[0] + w, top_left[1] + h)
    aligned_img = test_img[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]
    
    if aligned_img.shape != template_img.shape:
        aligned_img = cv2.resize(aligned_img, (template_img.shape[1], template_img.shape[0]))
    
    return aligned_img, None