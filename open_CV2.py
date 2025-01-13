import cv2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread("/Users/adityasoni234/Documents/MONAL/IMG_5337.JPG", cv2.IMREAD_COLOR)

plt.imshow("image",img)

cv2.waitKey(0)

cv2.destroyAllWindows()

h,w = img.shape[:2]
print("height",h)
print("width",w)

'''
height 2208
width 1242
'''

(B ,G ,R) = img[100,100]
print("BLUE",B)
print("GREEN",G)
print("RED",R)

'''
BLUE 66
GREEN 99
RED 168
'''
