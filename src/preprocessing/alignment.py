import cv2
import numpy as np

def align_images(test_img, template_img):
    """ORB feature alignment"""
    if len(test_img.shape) == 3:
        test_gray = cv2.cvtColor(test_img, cv2.COLOR_BGR2GRAY)
    else:
        test_gray = test_img.copy()
        
    if len(template_img.shape) == 3:
        template_gray = cv2.cvtColor(template_img, cv2.COLOR_BGR2GRAY)
    else:
        template_gray = template_img.copy()
    
    orb = cv2.ORB_create(5000)
    kp1, des1 = orb.detectAndCompute(test_gray, None)
    kp2, des2 = orb.detectAndCompute(template_gray, None)
    
    if des1 is None or des2 is None:
        print("Not enough features for alignment")
        return None, None
    
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    
    if not matches:
        print("No matches found for alignment")
        return None, None
    
    matches = sorted(matches, key=lambda x: x.distance)
    points1 = np.zeros((len(matches), 2), dtype=np.float32)
    points2 = np.zeros((len(matches), 2), dtype=np.float32)
    
    for i, match in enumerate(matches):
        points1[i, :] = kp1[match.queryIdx].pt
        points2[i, :] = kp2[match.trainIdx].pt
    
    if len(matches) >= 4:
        H, mask = cv2.findHomography(points1, points2, cv2.RANSAC, 5.0)
        
        if H is not None:
            height, width = template_gray.shape
            aligned_img = cv2.warpPerspective(test_img, H, (width, height))
            return aligned_img, H
    
    print("failed")
    return None, None



def simple_alignment(test_img, template_img):
    """Template matching alignment"""
    if len(test_img.shape) == 3:
        test_gray = cv2.cvtColor(test_img, cv2.COLOR_BGR2GRAY)
    else:
        test_gray = test_img.copy()
        
    if len(template_img.shape) == 3:
        template_gray = cv2.cvtColor(template_img, cv2.COLOR_BGR2GRAY)
    else:
        template_gray = template_img.copy()
    
    h, w = template_gray.shape
    result = cv2.matchTemplate(test_gray, template_gray, cv2.TM_CCOEFF_NORMED)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
    top_left = max_loc
    bottom_right = (top_left[0] + w, top_left[1] + h)
    aligned_img = test_img[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]
    if aligned_img.shape != template_img.shape:
        aligned_img = cv2.resize(aligned_img, (template_img.shape[1], template_img.shape[0]))
    
    return aligned_img, None