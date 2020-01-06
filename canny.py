import cv2
import numpy as np

def canny(image):
    gray = cv2.cvtColor(lane_image,cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray,(5,5),3)
    canny = cv2.Canny(blur,100,165)
    return canny

def display_lines(image,lines):
    line_image = np.zeros_like(image)
    if lines is not None:
        for line in lines:
            x1,y1,x2,y2 = line.reshape(4)
            cv2.line(line_image,(x1,y1),(x2,y2),(22,100,8),5)
    return line_image

image = cv2.imread("test_image.jpg")
lane_image = np.copy(image)
canny = canny(lane_image)
lines = cv2.HoughLinesP(canny,2,np.pi/180,100,np.array([]),minLineLength=18,maxLineGap=1)
line_image = display_lines(lane_image,lines)
cv2.imshow("Canny",line_image)
cv2.waitKey(0)