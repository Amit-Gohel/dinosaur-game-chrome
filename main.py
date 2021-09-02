import Hand_Tracking_Modual as htm
import cv2
import math
from pynput.keyboard import Key, Controller

keybord = Controller()

cap = cv2.VideoCapture(1)
cap.set(3, 1280)
cap.set(4, 1024)
detector = htm.handDetector(detectioCon=0.6, maxHands=1)

while True:
    sucesses, img = cap.read()
    img = detector.findHands(img)
    lmList = detector.findPositon(img, draw=False)
    if len(lmList) != 0:
        x1, y1 = lmList[4][1], lmList[4][2]
        x2, y2 = lmList[8][1], lmList[8][2]

        lenght = math.hypot(x2 - x1, y2 - y1)
        if lenght > 100:
            keybord.press(Key.space)
            keybord.release(Key.space)

    cv2.imshow("image", img)
    cv2.waitKey(1)
