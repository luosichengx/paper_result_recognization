import cv2
import predict
from constants import *
from mnist import *

im = cv2.imread(TEST_ID_BLOCK_FILE)


def proc(im):
    recognized_result = ''
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    ## 阈值分割
    ret, thresh = cv2.threshold(gray, 200, 255, 1)
    ## 对二值图像执行膨胀操作
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (5, 5))
    dilated = cv2.dilate(thresh, kernel)
    #################      Now finding Contours         ###################
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    ## 提取小方格，其父轮廓都为0号轮廓
    boxes = []
    l = len(hierarchy[0])
    for i in range(len(hierarchy[0])):
        if hierarchy[0][i][3] == 0:
            boxes.append(hierarchy[0][i])
    temp = {}
    for j in range(len(boxes)):
        if boxes[j][2] != -1:
            # number_boxes.append(boxes[j])
            x, y, w, h = cv2.boundingRect(contours[boxes[j][2]])

            # debug
            # im = cv2.rectangle(im,(x-1,y-1),(x+w+1,y+h+1),(0,0,255),2)
            # cv2.imshow("test",im)
            # cv2.waitKey()
            # print(x)


            temp[str(x)] = im[y:y + h, x:x + w]

    str_key_list = list(temp.keys())
    new_list = []
    for i in range(0, len(str_key_list)):
        new_list.append(int(str_key_list[i]))

    new_list.sort()
    print("new_list," ,new_list)
    for i in new_list:
        digit_pic = str(i) + ".png"
        pic_data = temp[str(i)]
        cv2.imwrite(digit_pic, pic_data)
        recognized_result += str(predict.predict_digital_image(digit_pic))

    return recognized_result


def recognize_test_id():
    return proc(cv2.imread(TEST_ID_BLOCK_FILE))


def recognize_stu_id():
    return proc(cv2.imread(STU_ID_BLOCK_FILE))


if __name__ == '__main__':
    print('试卷编号：',recognize_test_id())
    # print('学号：',recognize_stu_id())
