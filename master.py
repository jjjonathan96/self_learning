import cv2 as cv
import numpy as np

class master():

    def process(img):
        ll = 130
        hh = 255
        lower = np.array([ll, ll, ll])
        upper = np.array([hh, hh, hh])
        img = cv.resize(img,(700,400))
        thresh = cv.inRange(img, lower, upper)
        cv.imshow('thresh',thresh)
        cv.imwrite('thresh'+'.png',thresh)
        kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (20, 20))
        morph = cv.morphologyEx(thresh, cv.MORPH_CLOSE, kernel)
        mask = 255 - morph
        segmented = cv.bitwise_and(img, img, mask=mask)
        cv.imshow('segmented',segmented)
        # cv.waitKey(10)
        cv.imwrite('segmented'+'.png',segmented)
        # black patch detection

        gray = cv.cvtColor(segmented, cv.COLOR_BGR2GRAY)
        ret, thresh = cv.threshold(gray, 130, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)
        binary = cv.adaptiveThreshold(gray, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY_INV, 131, 15)
        _, _, boxes, _ = cv.connectedComponentsWithStats(binary)
        x1,y1,w1,h1,pixel1 = boxes[0]
        boxes = boxes[1:]
        filtered_boxes = []
        total_area = 0.0
        for x, y, w, h, pixels in boxes:
            if pixels < 8000 and h >6 and w > 6 and h < 700 and w < 700:
                # pixels < 1500 and h < 200 and w < 200 and
                filtered_boxes.append((x, y, w, h))
                # total_area = pixels
        for x, y, w, h in filtered_boxes:
            cv.rectangle(segmented, (x, y), (x + w, y + h), (0, 0, 255), 2)


        cv.imwrite('result' + ".png", segmented)