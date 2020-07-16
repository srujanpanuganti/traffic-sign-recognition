import numpy as np
import cv2
import sys
from numpy import interp
import math
# import train_red
from sklearn import svm
from PIL import Image
import recognize
np.set_printoptions(threshold=sys.maxsize)

import glob
import imutils

def get_red_contours(img_ihls):
    img_ihls[:,:,2] = cv2.equalizeHist(img_ihls[:,:,2])
    img_bgr = cv2.cvtColor(img_ihls,cv2.COLOR_HLS2BGR_FULL)

    image_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

    lower_red_1 = np.array([0,70,60])
    upper_red_1 = np.array([10,255,255])
    mask_1 = cv2.inRange(image_hsv,lower_red_1,upper_red_1)
    lower_red_2 = np.array([170,70,100])
    upper_red_2 = np.array([180,255,255])

    mask_2 = cv2.inRange(image_hsv,lower_red_2,upper_red_2)
    mask = cv2.bitwise_or(mask_1,mask_2)

    red_mask_ = cv2.bitwise_and(img_bgr,img_bgr, mask = mask)

    l_channel = red_mask_[:,:,2]
    s_channel = red_mask_[:,:,1]
    h_channel = red_mask_[:,:,0]

    filtered_r = cv2.medianBlur(l_channel,5)
    filtered_g = cv2.medianBlur(s_channel,5)
    filtered_b = cv2.medianBlur(h_channel,5)

    filtered_red = 10*filtered_r - 0*filtered_b + 0*filtered_g

    regions, _ = mser_red.detectRegions(np.uint8(filtered_red))
    hulls = [cv2.convexHull(p.reshape(-1, 1, 2)) for p in regions]


    blank= np.zeros_like(red_mask_)

    # hulls_ = []
    # if hulls != []:
    #     for hu in hulls:
    #         x,y,w,h = cv2.boundingRect(hu)
    #         #print(x)
    #         if(x<800):continue
    #         else:
    #             hulls_.append(hu)
    #

    cv2.fillPoly(np.uint8(blank), hulls, (0,0,255))

    kernel_2 = np.ones((5,5),np.uint8)

    opening = cv2.morphologyEx(blank, cv2.MORPH_OPEN, kernel_2)
    resized_eroded = cv2.resize(opening, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_CUBIC)

    _, r_thresh = cv2.threshold(opening[:,:,2], 60, 255, cv2.THRESH_BINARY)

    small_blank = np.zeros((64,64))

    cnts = cv2.findContours(r_thresh, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)

    return cnts , hulls

def get_red_predicted_image(cnts,frame,hulls,classifier):
    max_cnts = 3
    pred_im = None
    x_ = None
    y_ = None

    prediction_list = []
    countor_list = []


    cnts_sorted = sorted(cnts, key=cv2.contourArea, reverse=True)
    # print('in red pred')

    if len(cnts_sorted)>max_cnts:
        cnts_sorted = cnts_sorted[:3]

    for c in cnts_sorted:
        # print('in for')
        x_,y_,w,h = cv2.boundingRect(c)
        if(x_<800):continue
        aspect_ratio = w/h
        if aspect_ratio<=0.3 or aspect_ratio>1.2:
            continue

        cv2.drawContours(frame, hulls, -1, (0, 255, 0), 2)
        mask = np.zeros_like(frame) # Create mask where white is what we want, black otherwise
        cv2.drawContours(mask, [c], -1, (255,255,255), -1) # Draw filled contour in mask
        out = np.zeros_like(frame) # Extract out the object and place into output image
        out[mask == 255] = frame[mask == 255]
        x,y,_ = np.where(mask==255)
        (topx, topy) = (np.min(x), np.min(y))
        (botx, boty) = (np.max(x), np.max(y))
        if np.abs(topx - botx) <= 25 or np.abs(topy - boty)<= 25:
            continue


        out = fin[topx:botx+1, topy:boty+1]
        out_resize = cv2.resize(out, (64,64), interpolation=cv2.INTER_CUBIC)



        pred = recognize.predict_img_red(out_resize, classifier)
        if pred not in no_sign:
            # pred_im = mapping[str(pred[-1])]
            prediction_list.append(pred[0])
            countor_list.append(c)

    # return pred_im, x_, y_
    return prediction_list,countor_list

def get_blue_countours(img_ihls):

    img_ihls[:,:,1] = cv2.equalizeHist(img_ihls[:,:,1])
    img_bgr = cv2.cvtColor(img_ihls,cv2.COLOR_HLS2BGR_FULL)

    image_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

    lower_blue_1 = np.array([94,127,20])
    upper_blue_1 = np.array([126,255,200])
    mask = cv2.inRange(image_hsv,lower_blue_1,upper_blue_1)

    blue_mask_ = cv2.bitwise_and(img_bgr,img_bgr, mask = mask)

    l_channel = blue_mask_[:,:,2]
    s_channel = blue_mask_[:,:,1]
    h_channel = blue_mask_[:,:,0]

    filtered_r = cv2.medianBlur(l_channel,5)
    filtered_g = cv2.medianBlur(s_channel,3)
    filtered_b = cv2.medianBlur(h_channel,3)

    filtered_red = 0*filtered_r + 10*filtered_b + 10*filtered_g
    regions, _ = mser_blue.detectRegions(np.uint8(filtered_red))
    hulls = [cv2.convexHull(p.reshape(-1, 1, 2)) for p in regions]

    blank= np.zeros_like(blue_mask_)

    cv2.fillPoly(np.uint8(blank), hulls, (0,0,255))


    kernel_2 = np.ones((5,5),np.uint8)
    opening = cv2.morphologyEx(blank, cv2.MORPH_OPEN, kernel_2)

    resized_eroded = cv2.resize(opening, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_CUBIC)

    _, r_thresh = cv2.threshold(opening[:,:,2], 30, 255, cv2.THRESH_BINARY)

    small_blank = np.zeros((64,64))

    cnts = cv2.findContours(r_thresh, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)

    return cnts,hulls

def get_blue_predicted_image(cnts,frame,hulls,classifier):
    max_cnts = 3
    # x_ ,y_ = 0
    pred_im = None
    x_  = None
    y_ = None

    prediction_list = []
    countor_list = []


    # print('in blue pred')
    cnts_sorted = sorted(cnts, key=cv2.contourArea, reverse=True)
    if len(cnts_sorted)>max_cnts:
        cnts_sorted = cnts_sorted[:3]

    for c in cnts_sorted:
        # print('in sorted333')
        x_ , y_, w, h = cv2.boundingRect(c)

        #print(x_,y_)

        if (x_<100):
            # print('1')
            continue
        if h<20:
            # print('2')
            continue
        if h>20 and h<40:
            # print('3')
            h = h+20
            w = 0.7*h

        aspect_ratio = w/h

        if aspect_ratio<=0.3 or aspect_ratio>1.2:
            # print('4')
            continue
        cv2.drawContours(frame, hulls, -1, (0, 255, 0), 2)
        mask = np.zeros_like(frame) # Create mask where white is what we want, black otherwise
        cv2.drawContours(mask, [c], -1, (255,255,255), -1) # Draw filled contour in mask
        out = np.zeros_like(frame) # Extract out the object and place into output image
        out[mask == 255] = frame[mask == 255]
        x,y,_ = np.where(mask==255)
        (topx, topy) = (np.min(x), np.min(y))
        (botx, boty) = (np.max(x), np.max(y))
        if np.abs(topx - botx) <= 25 or np.abs(topy - boty)<= 25:
            # print('5')
            continue


        out = fin[topx:botx+1, topy:boty+1]
        out_resize = cv2.resize(out, (64,64), interpolation=cv2.INTER_CUBIC)

        pred, prob = recognize.predict_img_blue(out_resize, classifier)
        if pred not in no_sign:
            # pred_im = mapping[str(pred[-1])]
            # prediction_list.append(pred[0])
            # countor_list.append(c)
            prediction_list.append(pred[0])
            countor_list.append(c)

    return prediction_list, countor_list


    # print("xy",x_,y_)
    # return pred_im, x_ ,y_

path_red = {"./Training/00001", "./Training/00014", "./Training/00017", "./Training/00019", "./Training/00021", "./Training/00066", "./Training/00067", "./Training/00068","./Training/00069"}
path_blue = {"./Training/00035", "./Training/00038", "./Training/00045", "./Training/00071", "./Training/00070"}

classifier_red = svm.SVC(gamma='scale', decision_function_shape= 'ovo',probability = True)
classifier_blue = svm.SVC(gamma='scale', decision_function_shape= 'ovo',probability = True)

samples = ['/Training/00001/00025_00001.ppm', "/Training/00014/00208_00000.ppm", "/Training/00017/00002_00000.ppm", "/Training/00019/00006_00000.ppm", "/Training/00021/00375_00000.ppm", "/Training/00035/00020_00000.ppm", "/Training/00038/00004_00000.ppm", "/Training/00045/00012_00000.ppm"]
labels = [1, 14, 17, 19, 21, 35, 38, 45]

no_sign = [66,67,68,69,70,71]

mapping={}

for i in range(len(labels)):
    mapping[str(labels[i])] = cv2.resize(np.asarray(Image.open('/home/srujan/PycharmProjects/enpm673_pr6' + samples[i])), (int(64), int(64)))

# cv2.imshow('akak',mapping[1])
# cv2.waitKey(0)


# print(lll)
red_classifier = recognize.train_imgs_red(path_red,classifier_red)
blue_classifier = recognize.train_imgs_blue(path_blue,classifier_blue)

mser_red = cv2.MSER_create(8,400,2000)
mser_blue = cv2.MSER_create(8,400,2000)

image_list = []

filenames = [img for img in glob.glob("input/*.jpg")]
filenames.sort()
for img in filenames:
    image_list.append(img)

final = []

# image_list = image_list[110:]
ct = 0
out_imges = []

for img in image_list:
    ct +=1
    image = np.uint8(cv2.imread(img))
    frame = image.copy()
    fin = image.copy()
    img_ihls = cv2.cvtColor(frame,cv2.COLOR_BGR2HLS_FULL)

    red_cnts, red_hulls = get_red_contours(img_ihls)
    blue_cnts, blue_hulls = get_blue_countours(img_ihls)

    if not red_cnts == []:
        red_pred_list , red_cnt_list = get_red_predicted_image(red_cnts, frame, red_hulls, red_classifier)

        if red_pred_list is not None:
            for i in range(len(red_cnt_list)):
                x,y,w,h = cv2.boundingRect(red_cnt_list[i])
                cv2.rectangle(fin, (x,y), (int(x+w) , int(y+h)) , (0,255,0),2)
                new_x = x - 64
                new_y = y + 64

                pred_im_red = mapping[str(red_pred_list[i])]
                fin[y:new_y, new_x:x] = pred_im_red


    if not blue_cnts == []:
        # blue_pred, blue_x, blue_y = get_blue_predicted_image(blue_cnts, frame, blue_hulls, blue_classifier)
        blue_pred_list , blue_cnt_list = get_blue_predicted_image(blue_cnts, frame, blue_hulls, blue_classifier)
        # cv2.imshow('blue pred',blue_pred)
        # cv2.waitKey(0)

        if blue_pred_list is not None:
            for i in range(len(blue_cnt_list)):
                x,y,w,h = cv2.boundingRect(blue_cnt_list[i])
                cv2.rectangle(fin, (x,y), (int(x+w) , int(y+h)) , (0,255,0),2)
                new_x_ = x - 64
                new_y_ = y + 64

                pred_im_blue = mapping[str(blue_pred_list[i])]
                fin[y:new_y_, new_x_:x] = pred_im_blue


    resized = cv2.resize(frame, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_CUBIC)

    # if ct == 50:
    #     break

    # cv2.imshow("fn", fin)
    out_imges.append(fin)
    final.append(fin)
    # cv2.waitKey(5)

#
# source = cv2.VideoWriter_fourcc(*'XVID')
# out = cv2.VideoWriter('blue_detect.avi', source, 5.0, (1628,1236))

i = 0
for im in final:
    print(i)
    # cv2.imshow("camera", im)
    cv2.imwrite('outputs/pic{:>05}.jpg'.format(i), im)
    i+=1

#
#
# for image in out_imgs:
#     # print(ct)
#     out.write(image)
#     cv2.waitKey(10)
# out.release()
