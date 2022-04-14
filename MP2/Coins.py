import cv2
import numpy as np
import os

colorBlue = (255, 255, 0)
colorGreen = (0, 255, 0)
colorRed = (0,0,255)
colorYellow = (0,255,255)

#####################

def findAll(img):
    img = cv2.medianBlur(img, 5)
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    edges = cv2.Canny(img_hsv, 350, 620, apertureSize = 5)
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, 90,
                                minLineLength=50, maxLineGap=5)

    lowx = lines[1][0][0]
    highx = lines[1][0][0]
    lowy = lines[0][0][1]
    highy = lines[0][0][1]
    for line in lines:
        if lowx > line[0][0]:
            lowx = line[0][0]
        if highx < line[0][0]:
            highx = line[0][0]
        if lowy > line[0][1]:
            lowy = line[0][1]
        if highy < line[0][1]:
            highy = line[0][1]


    cv2.line(img,(lowx,lowy),(highx,lowy),colorYellow,5)
    cv2.line(img,(lowx,lowy),(lowx,highy),colorYellow,5)
    cv2.line(img,(highx,highy),(highx,lowy),colorYellow,5)
    cv2.line(img,(highx,highy),(lowx,highy),colorYellow,5)

    #####################


    circles = cv2.HoughCircles(img_hsv, cv2.HOUGH_GRADIENT, 1, 10, param1=100, param2=35, minRadius=20, maxRadius=40)
    circles = np.uint16(np.around(circles))

    arrayOfRadiuses = np.transpose(circles[0])[2]
    maxRadius = max(arrayOfRadiuses)

    countOfBigCoinsInside = 0
    countOfBigCoinsOutside = 0
    countOfSmallCoinsInside = 0
    countOfSmallCoinsOutside = 0

    for i in circles[0, :]:
        
        if i[0]>lowx and i[0]<highx and i[1]>lowy and i[1]<highy:
            if i[2] > maxRadius - 3:
                countOfBigCoinsInside += 1
                cv2.circle(img, (i[0], i[1]), i[2], colorBlue, 2)
            else:
                countOfSmallCoinsInside += 1
                cv2.circle(img, (i[0], i[1]), i[2], colorGreen, 2)
            cv2.circle(img,(i[0],i[1]),2,colorRed,3) 
        else:
            if i[2] > maxRadius - 3:
                countOfBigCoinsOutside += 1
                cv2.circle(img, (i[0], i[1]), i[2], colorBlue, 2)
            else:
                countOfSmallCoinsOutside += 1
                cv2.circle(img, (i[0], i[1]), i[2], colorGreen, 2)
            

    allMoneyInside = round(countOfBigCoinsInside * 5 + countOfSmallCoinsInside * 0.05, 2)
    allMoneyOutside = round(countOfBigCoinsOutside * 5 + countOfSmallCoinsOutside * 0.05, 2)
    allMoney = round(allMoneyInside + allMoneyOutside, 2)

    ###################

    print('Count of big coins inside ', countOfBigCoinsInside)
    print('Count of big coins outside ', countOfBigCoinsOutside)
    print('Count of small coins inside ', countOfSmallCoinsInside)
    print('Count of small coins outside ', countOfSmallCoinsOutside)
    print('All money inside ', allMoneyInside)
    print('All money outside ', allMoneyOutside)
    print('All money ', allMoney)

    cv2.imshow('detected circles', img)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

def main():
    DIRECTORY = 'WMA/MP2/resourses/'
    coins_images = []

    for entry in os.scandir(DIRECTORY):
        if entry.path.endswith('.jpg') and entry.is_file():
            try:
                img_data = cv2.imread(entry.path)
                coins_images.append(img_data)
            except Exception:
                pass

    for image in coins_images:
        findAll(image)

if __name__ == '__main__':
    main()
