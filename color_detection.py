import cv2
import numpy as np

cap = cv2.VideoCapture(0)

while True:
    success, frame = cap.read()
    if not success:
        break

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)


    lower_red1 = np.array([0, 100, 100])     # First red range (0-10)
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([160, 100, 100])   # Second red range (160-180)
    upper_red2 = np.array([180, 255, 255])


    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask = mask1 + mask2

    lower_blue = np.array([100, 100, 100])  # H: 100 is around blue
    upper_blue = np.array([130, 255, 255])  # H: 130 covers the blue range

    # Create mask for blue
    mask3 = cv2.inRange(hsv, lower_blue, upper_blue)

    mask = mask1 + mask2 +mask3

    # Combine the masks


    result = cv2.bitwise_and(frame, frame, mask=mask)

    # cv2.imshow("Frame", frame)
    cv2.imshow("result", result)


    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()