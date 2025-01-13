import cv2

cap = cv2.VideoCapture(0)

while True:
    success, img = cap.read()
    if not success:
        break

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    edges = cv2.Canny(img, 100, 200)
    cv2.imshow("Edges", edges)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
