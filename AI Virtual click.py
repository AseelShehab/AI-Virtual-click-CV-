import cv2
import autopy
import numpy as np
import Handtrack_class as htm

##########################
wCam, hCam = 640, 480   # أبعاد الكاميرا
frameR = 100            # منطقة الإطار (تقليل المساحة للحركة)
smoothening = 7         # لتنعيم حركة الماوس
##########################

pLocX, pLocY = 0, 0     # الموقع السابق للماوس
cLocX, cLocY = 0, 0     # الموقع الحالي للماوس

cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)

detector = htm.handDetector(maxHands=1)
wScr, hScr = autopy.screen.size()  # أبعاد الشاشة

while True:
    # 1. اقرأ الصورة من الكاميرا
    success, img = cap.read()
    if not success:
        break
    img = cv2.flip(img, 1)  # تقليب الصورة للمرآة

    # 2. اكتشف اليد
    img = detector.findHands(img)
    lmList, bbox = detector.findPosition(img, draw=False)

    if len(lmList) != 0:
        # 3. إحداثيات السبابة والوسطى
        x1, y1 = lmList[8][1:]
        x2, y2 = lmList[12][1:]

        # 4. أي أصابع مرفوعة؟
        fingers = detector.fingersUp()

        # 5. وضعية التحريك: إذا السبابة فقط مرفوعة
        if fingers[1] == 1 and fingers[2] == 0:
            # تحويل إحداثيات الكاميرا لإحداثيات الشاشة
            x3 = np.interp(x1, (frameR, wCam - frameR), (0, wScr))
            y3 = np.interp(y1, (frameR, hCam - frameR), (0, hScr))

            # تنعيم الحركة
            cLocX = pLocX + (x3 - pLocX) / smoothening
            cLocY = pLocY + (y3 - pLocY) / smoothening

            autopy.mouse.move(wScr - cLocX, cLocY)
            cv2.circle(img, (x1, y1), 15, (255, 0, 255), cv2.FILLED)

            pLocX, pLocY = cLocX, cLocY

        # 6. وضعية الكليك: إذا السبابة والوسطى مرفوعين
        if fingers[1] == 1 and fingers[2] == 1:
            length, img, lineInfo = detector.findDistance(8, 12, img)
            if length < 40:  # إذا الأصبعين قريبين
                cv2.circle(img, (lineInfo[4], lineInfo[5]),
                           15, (0, 255, 0), cv2.FILLED)
                autopy.mouse.click()

    # 7. عرض الصورة
    cv2.imshow("Image", img)

    # 8. الخروج عند الضغط على q
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
