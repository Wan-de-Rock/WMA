import cv2
from cv2 import rectangle
from cv2 import imshow
import numpy as np

def main():
    ball = cv2.imread("WMA/MP4/ball.png")
    

    cap = cv2.VideoCapture(0)
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_frontalface_alt2.xml')
    scaling_factor = 0.5
    while True:
        ret, frame = cap.read()
        if ret:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(
                gray, scaleFactor=1.05, minNeighbors=5, minSize=(50, 50))

            if len(faces) > 0:
                for face in faces:

                    x, y, w, h = face

                    x_center = (int)(x + w/2)
                    y_center = (int)(y + h/2)

                    w_mask = (int)(w/4)
                    h_mask = (int)(h/4)

                    x_mask = (int)(x_center - w_mask/2)
                    y_mask = (int)(y_center - h_mask/2)

                    ballCopy = ball.copy()
                    ballCopy = cv2.resize(ballCopy, (h_mask, w_mask), interpolation=cv2.INTER_LINEAR)
                    frame[y_mask:y_mask+h_mask, x_mask:x_mask+w_mask] = ballCopy

                    #rectangle(frame, (x,y), (x+w, y+h), (0,255,0),2)

        

        cv2.imshow('Video',frame)
        key = cv2.waitKey(1)
        if key == 27:
            cv2.destroyAllWindows()
            break


if __name__ == '__main__':
    main()