import cv2
import numpy as np
from image_task import objectFinder

video = cv2.VideoCapture()
video.open('WMA/MP1/resourses/movingball.mp4')

frame_width = int(video.get(3))
frame_height = int(video.get(4))
size = (frame_width, frame_height)
result = cv2.VideoWriter(
    'WMA/MP1/result.avi',  cv2.VideoWriter_fourcc(*'MJPG'), 20, size)


while True:
    success, frame_rgb = video.read()
    if not success:
        break   
    result.write(objectFinder(frame_rgb))

video.release()
#result.release()