import cv2

cap = cv2.VideoCapture(0)

success,prev_frame = cap.read()
#prev_frame_gray = cv2.cvtColor(prev_frame,cv2.COLOR_BGR2GRAY)

while True:

    success,frame = cap.read()

    if not success:
        break

# current_frame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

    diff = cv2.absdiff(prev_frame,frame)

    _,thresold = cv2.threshold(diff,30,255,cv2.THRESH_BINARY)

    cv2.imshow("Motion Detection",thresold)

    prev_frame = frame

    if cv2.waitKey(1) & 0xFF == ord("x"):
        break

cap.release()
cap.destroyAllWindows()