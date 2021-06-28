import cv2
import os
import numpy as np
mix_pic = 'D:/finisg_mixpic'
labels = 'D:/bboxs'


for i in os.listdir(labels):
    lab_path = os.path.join(labels,i)
    with open(lab_path,'r') as f:
        # txt = f.readline().strip()
        txt = f.readline()
        # print(txt)
        str_boxlist = txt.split(' ')[1:]
        bbox = [float(x) for x in str_boxlist]
        # print(bbox)
    name = i.split('.')[0]
    img_path = os.path.join(mix_pic,name+'.png')
    print(img_path)
    img = cv2.imread(img_path)
    h,w,c = img.shape
    assert img is not None
    xmax = int((2 * bbox[0] * w + bbox[2] * w) / 2)
    xmin = int((2 * bbox[0] * w - bbox[2] * w) / 2)
    ymax = int((2 * bbox[1] * h + bbox[3] * h) / 2)
    ymin = int((2 * bbox[1] * h - bbox[3] * h) / 2)
    cv2.rectangle(img,(xmin,ymin),(xmax,ymax),(0,255,0),2)
    print(img)
    cv2.imshow('beatiful',img)
    cv2.waitKey(100)