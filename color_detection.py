import cv2
import numpy as np

cap = cv2.VideoCapture(0)

while True:
    success, frame = cap.read()
    if not success:
        break

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)


    lower_red = np.array([161, 155, 84])
    upper_red = np.array([179, 255, 255])

    mask2 = cv2.