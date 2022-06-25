import time

import cv2 as cv
import numpy as np
from imutils.video import FPS


cap = cv.VideoCapture(2)

# To get number of frames
n_frames = int(cap.get(cv.CAP_PROP_FRAME_COUNT))

# To check the number of frames in the video
print(n_frames)


width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))

print("height", width)
print("height", height)

count = 0
if not (cap.isOpened()):

    print('Could not open video device')
#https://www.e-consystems.com/blog/camera/technology/how-to-access-cameras-using-opencv-with-python/
cap.set(cv.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv.CAP_PROP_FRAME_HEIGHT, 480)

def process(img):
    ll = 2
    hh = 20
    lower = np.array([ll, ll, ll])
    upper = np.array([hh, hh, hh])
    thresh = cv.inRange(img, lower, upper)
    # cv.imshow('thresh',thresh)

    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (20, 20))
    morph = cv.morphologyEx(thresh, cv.MORPH_CLOSE, kernel)
    #cv.imshow('morph',morph)
    mask = 255 - morph
    #cv.imshow('mask',mask)
    segmented = cv.bitwise_and(img, img, mask=mask)
    #cv.imshow('segmented',segmented)
#comtour
    gray = cv.cvtColor(segmented, cv.COLOR_BGR2GRAY)
    ret, thresh = cv.threshold(gray, 130, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)
    binary = cv.adaptiveThreshold(gray, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY_INV, 131, 15)
    _, _, boxes, _ = cv.connectedComponentsWithStats(binary)
    x1, y1, w1, h1, pixel1 = boxes[0]
    areaOfSegmented = w1 * h1

    boxes = boxes[1:]
    filtered_boxes = []
    total_area = 0.0
    for x, y, w, h, pixels in boxes:
        if pixels < 6000 and h > 6 and w > 6 and h < 600 and w < 500:
            # pixels < 1500 and h < 200 and w < 200 and
            filtered_boxes.append((x, y, w, h))
            # total_area = pixels
    for x, y, w, h in filtered_boxes:

        if x>x1 and x<x1+w1:
            if y>y1 and y<y1+h:
                cv.rectangle(segmented, (x, y), (x + w, y + h), (0, 0, 255), 2)
                total_area += w * h

    org = (50, 50)
    font = cv.FONT_HERSHEY_SIMPLEX
    fontScale = 1
    color2 = (0, 255, 0)
    thickness = 2
    # print('total area', total_area)
    ratio = ((total_area) / (areaOfSegmented)) * 100
    ratio = np.round(ratio,2)
    #print(ratio)

    if ratio < 0.3:
        pre = 'class 1'
    elif ratio < 1.5:
        pre = 'class 2'
    else:
        pre = 'class 3'

    print(pre)
    cv.putText(segmented,'ratio'+ str(ratio) +' '+'predicted '+pre , (50, 50), font, fontScale, color2, thickness, cv.LINE_AA, False)
    # cv.imwrite('result' + ".png", segmented)
    #cv.imshow('result', segmented)
    hori = np.concatenate((img,segmented),axis=1)
    verti = np.concatenate((thresh,morph,mask),axis = 1)
    #thresh,morph,mask,
    cv.imshow('Quality', hori)
    # cv.imshow('canny process', verti)
    cv.waitKey(100)

def removeReflection(img):

    clahefilter = cv.createCLAHE(clipLimit=2.0, tileGridSize=(16, 16))



    while True:
        t1 = time.time()
        img = img.copy()

        ## crop if required
        # FACE
        x, y, h, w = 550, 250, 400, 300
        # img = img[y:y+h, x:x+w]

        # NORMAL
        # convert to gray
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        grayimg = gray

        GLARE_MIN = np.array([0, 0, 50], np.uint8)
        GLARE_MAX = np.array([0, 0, 225], np.uint8)

        hsv_img = cv.cvtColor(img, cv.COLOR_BGR2HSV)

        # HSV
        frame_threshed = cv.inRange(hsv_img, GLARE_MIN, GLARE_MAX)

        # INPAINT
        mask1 = cv.threshold(grayimg, 220, 255, cv.THRESH_BINARY)[1]
        result1 = cv.inpaint(img, mask1, 0.1, cv.INPAINT_TELEA)

        # CLAHE
        claheCorrecttedFrame = clahefilter.apply(grayimg)

        # COLOR
        lab = cv.cvtColor(img, cv.COLOR_BGR2LAB)
        lab_planes = cv.split(lab)
        clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        lab_planes[0] = clahe.apply(lab_planes[0])
        lab = cv.merge(lab_planes)
        clahe_bgr = cv.cvtColor(lab, cv.COLOR_LAB2BGR)

        # INPAINT + HSV
        result = cv.inpaint(img, frame_threshed, 0.1, cv.INPAINT_TELEA)

        # INPAINT + CLAHE
        grayimg1 = cv.cvtColor(clahe_bgr, cv.COLOR_BGR2GRAY)
        mask2 = cv.threshold(grayimg1, 220, 255, cv.THRESH_BINARY)[1]
        result2 = cv.inpaint(img, mask2, 0.1, cv.INPAINT_TELEA)

        # HSV+ INPAINT + CLAHE
        lab1 = cv.cvtColor(result, cv.COLOR_BGR2LAB)
        lab_planes1 = cv.split(lab1)
        clahe1 = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        lab_planes1[0] = clahe1.apply(lab_planes1[0])
        lab1 = cv.merge(lab_planes1)
        clahe_bgr1 = cv.cvtColor(lab1, cv.COLOR_LAB2BGR)

        # fps = 1./(time.time()-t1)
        # cv2.putText(clahe_bgr1    , "FPS: {:.2f}".format(fps), (10, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255))

        # display it
        # cv.imshow("IMAGE", img)
        # cv.imshow("GRAY", gray)
        # cv.imshow("HSV", frame_threshed)
        # cv.imshow("CLAHE", clahe_bgr)
        # cv.imshow("LAB", lab)
        # cv.imshow("HSV + INPAINT", result)
        # cv.imshow("INPAINT", result1)
        # cv.imshow("CLAHE + INPAINT", result2)
        # cv.imshow("HSV + INPAINT + CLAHE   ", clahe_bgr1)

        hori = np.concatenate((gray,frame_threshed),axis = 1)
        hori1 = np.concatenate((clahe_bgr,lab,result,result1,result2,clahe_bgr1),axis = 1)
        cv.imshow('result',hori)
        cv.imshow('result1',hori1)
        cv.waitKey(10000)
        # Break with esc key
        if cv.waitKey(1) & 0xFF == ord('q'):
            break

        cv.destroyAllWindows()


def brightnessRemove(img):
    array = np.array(img)
    print(array.shape)

    for k in range(3):
        for i in range(400):
            for j in range(500):
                print(array[i][j][k])
                



while True:
    success, img = cap.read(0)




    if success:

        img = cv.resize(img, (500, 400), cv.INTER_LINEAR)
        #cv.imwrite(str(count)+'.png',img)
        #count += 1


        #cv.imshow("Webcam", img)
        noiseRemoved = cv.fastNlMeansDenoisingColored(img)
        #brightnessRemove(noiseRemoved)
        #cv.imshow('Removed noise', noiseRemoved)
        process(noiseRemoved)
        #removeReflection(noiseRemoved)


        if cv.waitKey(1) == ord('q'):
            break

