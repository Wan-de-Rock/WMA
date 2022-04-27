from operator import itemgetter
import cv2
from cv2 import destroyAllWindows
import numpy as np
from matplotlib import pyplot as plt


def find(image_template, video): #reference_image
    paused = True
    recieved, frame = video.read()

    sift = cv2.xfeatures2d.SIFT_create()
    matcher = cv2.FlannBasedMatcher(dict(algorithm=1, trees=5))

    gray_template = cv2.cvtColor(image_template, cv2.COLOR_BGR2GRAY) #reference_image_bw
    keypoints_template, descriptors_template = sift.detectAndCompute(gray_template, None)

    while recieved:
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) #reference_image_bw
        keypoints_frame, descriptors_frame = sift.detectAndCompute(gray_frame, None)

        matches = matcher.match(descriptors_template, descriptors_frame)
        matches = sorted(matches, key=lambda x: x.distance)
        matches = [match.trainIdx for match in matches[:9]]

        matched_keypoints = np.float32([keypoint.pt for keypoint in itemgetter(*matches)(keypoints_frame)])

        rect = cv2.boundingRect(matched_keypoints)
        cv2.rectangle(frame, rect, [0,0,255])

        key = cv2.waitKey(2)
        if key == ord('q'):
            break

        cv2.imshow('Video', frame)

        recieved, frame = video.read()
    cv2.destroyAllWindows()











def main():
    cap = cv2.VideoCapture('WMA/MP3/resourses/sawmovie.mp4')
    #videopath = 'WMA/MP3/resourses/sawmovie.mp4'
    image_template = cv2.imread('WMA/MP3/resourses/saw1.jpg')
    #readVideo(videopath, image_template)
    find(image_template, cap)


if __name__=="__main__":
    main()
