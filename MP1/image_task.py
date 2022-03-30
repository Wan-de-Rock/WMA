import cv2
import numpy as np


i = cv2.imread("WMA/MP1/resourses/ball.png")

def objectFinder(img):

    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    low_red = np.array((0, 55, 55), np.uint8)
    high_red = np.array((5, 255, 255), np.uint8)

    mask_red = cv2.inRange(img_hsv, low_red, high_red)

    kernel = np.ones((4, 4), np.uint8)
    mask_without_noise = cv2.morphologyEx(mask_red, cv2.MORPH_OPEN, kernel)
    mask_closed = cv2.morphologyEx(mask_without_noise, cv2.MORPH_CLOSE, kernel)
    
    #res1 = cv2.bitwise_and(img_hsv, img, mask=mask_without_noise) 
    res = cv2.medianBlur(mask_closed, ksize=5) 

    #result = cv2.bitwise_and(img_hsv, img_hsv, mask=mask_without_noise)

    #contours, hierarchy = cv2.findContours(res, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    M = cv2.moments(res, 1)
    #M = cv2.moments(contours[0])
    cx = int(M['m10'] / M['m00'])
    cy = int(M['m01'] / M['m00'])

    cv2.drawMarker(img, (int(cx), int(cy)), color=(0, 255, 0), markerType=cv2.MARKER_CROSS, thickness=2)
    return img


def main():
    cv2.imshow("result", objectFinder(i))
    cv2.waitKey(0)

if __name__ == '__main__':
    main()