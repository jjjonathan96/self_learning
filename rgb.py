import cv2
import numpy



def detectBlackPatch(img):
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    binary_img = cv2.adaptiveThreshold(gray_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY_INV, 131, 15)

    _, _, boxes, _ = cv2.connectedComponentsWithStats(binary_img)
    # first box is the background
    boxes = boxes[1:]
    filtered_boxes = []
    for x,y,w,h,pixels in boxes:
        if pixels < 10000 and h < 200 and w < 200 and h > 10 and w > 10:
            filtered_boxes.append((x,y,w,h))

    for x,y,w,h in filtered_boxes:
        cv2.rectangle(img, (x,y), (x+w,y+h), (0,0,255),2)
    return img


def removeBackground(img):
    ll = 0
    hh = 5
    lower = numpy.array([ll, ll, ll])
    upper = numpy.array([hh, hh, hh])
    thresh = cv2.inRange(img, lower, upper)
    # cv.imshow('thresh',thresh)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (20, 20))
    morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    # cv.imshow('morph',morph)
    mask = 255 - morph
    # cv.imshow('mask',mask)
    segmented = cv2.bitwise_and(img, img, mask=mask)
    # cv.imshow('segmented',segmented)
    # comtour
    gray = cv2.cvtColor(segmented, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray, ll, hh, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 131, 15)
    _, _, boxes, _ = cv2.connectedComponentsWithStats(binary)
    x1, y1, w1, h1, pixel1 = boxes[0]
    areaOfSegmented = w1 * h1
    print(areaOfSegmented)
    boxes = boxes[1:]
    filtered_boxes = []
    total_area = 0.0
    for x, y, w, h, pixels in boxes:
        if pixels < 6000 and h > 6 and w > 6 and h < 600 and w < 500:
            # pixels < 1500 and h < 200 and w < 200 and
            filtered_boxes.append((x, y, w, h))
            # total_area = pixels
    for x, y, w, h in filtered_boxes:

        # if x > x1 and x < x1 + w1:
        #     if y > y1 and y < y1 + h:
        cv2.rectangle(segmented, (x, y), (x + w, y + h), (0, 0, 255), 2)
        total_area += w * h

    org = (50, 50)
    font = cv2.FONT_HERSHEY_SIMPLEX
    fontScale = 1
    color2 = (0, 255, 0)
    thickness = 2
    # print('total area', total_area)
    ratio = ((total_area) / (areaOfSegmented)) * 100
    ratio = numpy.round(ratio, 2)
    # print(ratio)

    if ratio < 0.3:
        pre = 'class 1'
    elif ratio < 1.5:
        pre = 'class 2'
    else:
        pre = 'class 3'

    print(pre)
    cv2.putText(segmented, 'ratio' + str(ratio) + ' ' + 'predicted ' + pre, (50, 50), font, fontScale, color2, thickness,
               cv2.LINE_AA, False)
    # # cv.imwrite('result' + ".png", segmented)
    # # cv.imshow('result', segmented)
    # return result
    return segmented


cap = cv2.VideoCapture(2)
count = 0
while True:
    success, img = cap.read(2)
    #cap.set(cv2.CAP_PROP_FPS, 10)

    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))  # depends on fourcc available camera
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 5)
    fps = int(cap.get(5))
    print("fps:", fps)

    if success:
        #cv2.imshow('captured',img)
        img = cv2.resize(img, (500, 400), cv2.INTER_LINEAR)


        #cv.imshow("Webcam", img)
        #img = noiseRemoved = cv2.fastNlMeansDenoisingColored(img)
        #brightnessRemove(noiseRemoved)
        #cv.imshow('Removed noise', noiseRemoved)


        img1 = img.copy()
        img1 = cv2.resize(img1, (600, 450))
        img2 = img1.copy()
        img3 = img1.copy()
        img = cv2.resize(img, (200, 150))
        blue, green, red = cv2.split(img)

        zeros = numpy.zeros(blue.shape, numpy.uint8)

        blueBGR = cv2.merge((blue, zeros, zeros))
        greenBGR = cv2.merge((zeros, green, zeros))
        redBGR = cv2.merge((zeros, zeros, red))

        # cv2.imshow('blue BGR', blueBGR)
        # cv2.imshow('green BGR', greenBGR)
        # cv2.imshow('red BGR', redBGR)
        #
        hori = numpy.concatenate((blueBGR, greenBGR, redBGR), axis=0)
        #
        bgr_planes = cv2.split(img)
        histSize = 256
        histRange = (0, 256)  # the upper boundary is exclusive
        accumulate = False
        b_hist = cv2.calcHist(bgr_planes, [0], None, [histSize], histRange, accumulate=accumulate)
        g_hist = cv2.calcHist(bgr_planes, [1], None, [histSize], histRange, accumulate=accumulate)
        r_hist = cv2.calcHist(bgr_planes, [2], None, [histSize], histRange, accumulate=accumulate)
        hist_w = 200
        hist_h = 150
        bin_w = int(round(hist_w / histSize))
        histImage = numpy.zeros((hist_h, hist_w, 3), dtype=numpy.uint8)
        cv2.normalize(b_hist, b_hist, alpha=0, beta=hist_h, norm_type=cv2.NORM_MINMAX)
        cv2.normalize(g_hist, g_hist, alpha=0, beta=hist_h, norm_type=cv2.NORM_MINMAX)
        cv2.normalize(r_hist, r_hist, alpha=0, beta=hist_h, norm_type=cv2.NORM_MINMAX)

        b = histImage.copy()
        g = histImage.copy()
        r = histImage.copy()

        for i in range(1, histSize):
            cv2.line(b, (bin_w * (i - 1), hist_h - int(b_hist[i - 1])),
                     (bin_w * (i), hist_h - int(b_hist[i])),
                     (255, 0, 0), thickness=2)
            cv2.line(g, (bin_w * (i - 1), hist_h - int(g_hist[i - 1])),
                     (bin_w * (i), hist_h - int(g_hist[i])),
                     (0, 255, 0), thickness=2)
            cv2.line(r, (bin_w * (i - 1), hist_h - int(r_hist[i - 1])),
                     (bin_w * (i), hist_h - int(r_hist[i])),
                     (0, 0, 255), thickness=2)

        # cv2.imshow('calcHist Demo', histImage)
        hori2 = numpy.concatenate((b, g, r), axis=0)
        segmented = removeBackground(img3)
        result = detectBlackPatch(segmented)
        verti = numpy.concatenate((img1, hori, hori2, result), axis=1)
        cv2.imshow('Quality', verti)
        cv2.imwrite(str(count)+'.png',verti)
        count += 1
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        if cv2.waitKey(1) == ord('q'):
            break


