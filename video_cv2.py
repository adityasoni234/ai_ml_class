import cv2
import numpy as np
import matplotlib.pyplot as plt

cap = cv2.VideoCapture(0)

while True:
    ret,frame = cap.read()
    
    if not ret:#if frame is not avilabel
        break
    
    cv2.imshow("frame",frame)

    if cv2.waitKey(1) & 0xFF == ord("x"):
        break

cap.release()
cv2.destroyAllWindows()