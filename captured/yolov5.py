import torch
import cv2 as cv
import os
files = os.listdir("C:/Users/johnn/Desktop/test/")
files_class1 = os.listdir("H:/dataset/class1")
files_class2 = os.listdir("H:/dataset/class2")
files_class3 = os.listdir("H:/dataset/class3")


#cap = cv.VideoCapture(2)
model = torch.hub.load('ultralytics/yolov5', 'custom', path='G:/Self Learning/model/yolov5.pt')


countClass3 = 0
for images in files_class2:

    img = cv.imread('H:/dataset/class2/'+ images)
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    result = model(img)
    result.show()
    result.save('H:/dataset/class2/'+"*" + images)
    tensor = result.xyxy[0]
    #print(tensor[0][5])

    # #print(len(tensor[0]))
    # for i in range(len(tensor)):
    #     # print(tensor[i][5])
    #     # print(type(tensor[i][5]))
    #     # print(int(tensor[i][5]))
    #
    #
    #     if 1 == int(tensor[i][5]):
    #         #result.show()
    #         if float(tensor[i][4]) >= 0.5:
    #             print('class 3')
    #             cv.waitKey(1000)
    #             break

    countMango = 0
    countBlackPatch = 0
    countChemicalPatch = 0
    patch = {}
    for  j in range(len(tensor)):
        if 0 == int(tensor[j][5]): #mango class
            x, y , h, w = tensor[j][0], tensor[j][1], tensor[j][2], tensor[j][3] #mango coordinates
            if tensor[j][3] >= 0.5: # confidence check of whether it is mango or not
                countMango += 1

        elif 1 ==  int(tensor[j][5]): #black patch classes

            countBlackPatch += 1
            x1, y1, h1, w1 = tensor[j][0], tensor[j][1], tensor[j][2], tensor[j][3]
            patchNew = {
                x:x1,
                y:y1,
                h:h1,
                w:w1

            }

            patch[str(countBlackPatch)] = patchNew
        elif 2 ==  int(tensor[j][5]):
            countChemicalPatch += 1
        else:
            pass
    print('mangoes',countMango)
    print('blackPatch', countBlackPatch)
    print('chemical patch',countChemicalPatch)

    if countMango == 1 and countBlackPatch ==0 and countChemicalPatch ==0:
        print('class 1')


    elif countMango == 1 and countBlackPatch >=0 and countChemicalPatch >=1:
        print('class 2')


    elif countMango == 1 and countBlackPatch >=1 and countChemicalPatch >=0:
        print('Class 3')
        countClass3 += 1

    else:
        print('unknown')

    print('#######################################################')

# while True:
#     success, img = cap.read(0)
#     #
#
#     if success:
#
#         result.show()
#         if cv.waitKey(1) == ord('q'):
#             break

print('class 2',countClass3)