
import cv2
import numpy as np


cap = cv2.VideoCapture(0)

lower_purple = np.array([-10, 100, 100])
upper_purple = np.array([10, 255, 255])

points = []


ret, frame = cap.read()
detect_frame = frame.copy()
Height, Width = frame.shape[:2]
frame_count = 0

while True:
    ret, frame = cap.read()
    hsv_img = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv_img, lower_purple, upper_purple)
    _, contours, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

#     contours, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    center =   int(Height/2), int(Width/2)

    if len(contours) > 0:
        c = max(contours, key=cv2.contourArea)
        (x, y), radius = cv2.minEnclosingCircle(c)
        M = cv2.moments(c)

        try:
            center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))

        except:
            center =   int(Height/2), int(Width/2)
        if radius > 25:
            cv2.circle(frame, (int(x), int(y)), int(radius),(0, 0, 255), 2)
            cv2.circle(frame, center, 5, (0, 255, 0), -1)
            detect_frame = frame.copy()

    points.append(center)

            

    frame = cv2.flip(frame, 1)
    cv2.imshow("Object Tracker", frame)

    if cv2.waitKey(1) == 27: 
        break


cap.release()
cv2.destroyAllWindows()
