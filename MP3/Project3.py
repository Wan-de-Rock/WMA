import os
from turtle import width

import cv2
import matplotlib.pyplot as plt
import numpy as np
from cv2 import bitwise_or
from screeninfo import get_monitors

def resize(img, s):
    image = img.copy()
    h, w = image.shape[:2]
    h = h + int(h*s)
    w = w + int(w*s)
    image = cv2.resize(image, (w, h), interpolation=cv2.INTER_LINEAR)
    return image

def norm_size(img):
    image = img.copy()
    screen = get_monitors()[0]
    height = screen.height
    width = screen.width-200
    h, w = image.shape[:2]
    if h > height:
        s = (1 - (height/h)) * (-1)
        image = resize(image, s)
    h, w = image.shape[:2]
    if w > width:
        s = (1 - (width/w)) * (-1)
        image = resize(image, s)
    return image

def findMatches(image):

    global saw_images

    img = saw_images[0]
    img1 = image

    grey1 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    grey2 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)

    grey1 = cv2.medianBlur(grey1, 7) ##########
    grey2 = cv2.medianBlur(grey2, 7) ##########

    siftobject = cv2.xfeatures2d.SIFT_create()

    keypoint1, descriptor1 = siftobject.detectAndCompute(grey1, None)
    keypoint2, descriptor2 = siftobject.detectAndCompute(grey2, None)

    keypointimage1 = cv2.drawKeypoints(img, keypoint1, img, color=(
        0, 255, 0), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    keypointimage2 = cv2.drawKeypoints(img1, keypoint1, img1, color=(
        0, 255, 0), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    matcher = cv2.BFMatcher(cv2.NORM_L1, crossCheck=True)
    matches = matcher.match(descriptor1, descriptor2)
    matches = sorted(matches, key=lambda x: x.distance)
    matched_img = cv2.drawMatches(img, keypoint1, img1, keypoint2, matches[:20], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)


    cv2.imshow('result', norm_size(matched_img))

    cv2.waitKey(0)
    cv2.destroyAllWindows()

def main():

    DIRECTORY = 'WMA/MP3/resourses/'
    global saw_images
    saw_images = []

    for entry in os.scandir(DIRECTORY):
        if entry.path.endswith('.jpg') and entry.is_file():
            try:
                img_data = cv2.imread(entry.path)
                saw_images.append(img_data)
            except Exception:
                pass

    for image in saw_images[1:]:
        findMatches(image)

if __name__ == '__main__':
    main()