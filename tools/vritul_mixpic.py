import os
import shutil

import cv2
import random
import numpy as np
style_pic = 'C:/Users/EDZ/Desktop/back_pic'
data_dir = 'D:/data_23/data_23/'
results = 'D:/mixing_pic'
mix_pic = 'D:/finisg_mixpic'
labels = 'D:/bboxs'
# img = cv2.imread(style_pic+'/q.jpeg')
# img = cv2.resize(img,(1024,1024))
# cv2.imwrite(style_pic+'/p.jpeg',img)
###  从原数据路径，文件夹 0-23，每个文件夹选取100张图片。复制到mixing_pic文件夹里面去。
# if not os.path.exists(results):
#     os.mkdir(results)
# folder_files = os.listdir(data_dir)
# for folder in folder_files:
#     need_path = os.path.join(data_dir,folder)
#     if os.path.isdir(need_path):
#         for i,img_name in enumerate(os.listdir(need_path)):
#             if i<100:
#                 img_path = os.path.join(need_path,img_name)
#                 shutil.copy(img_path,results)

sty_files = os.listdir(style_pic)
for img_name in os.listdir(results):
        img_path = os.path.join(results,img_name)
        # name = img_name.split('.')[0]
        img = cv2.imread(img_path)
        h,w,c = img.shape
        # img = cv2.resize(img,(512,512))
        styimg_name = random.choice(sty_files)
        styimg_path = os.path.join(style_pic,styimg_name)
        styimg = cv2.imread(styimg_path)

        h1,w1,c1 = styimg.shape
        styimg = cv2.resize(styimg,(int(0.7*h1),int(0.7*w1)))
        h1, w1, c1 = styimg.shape
        center_x,center_y = int(h1/2),int(w1/2)
        b = np.arange(0,350)

        c_x0 = random.choice(b)
        c_y0 = random.choice(b)

        print(styimg_name)
        print(h1,c_x0,h1-c_x0,h,w1,c_y0,w1-c_y0,w)
        # print()
        syt_txt = img_name.split('.')[0]
        gqk_name = os.path.join('/home/meprint/guoqk/data/sun/', img_name)
        if h1-c_x0>=h and w1-c_y0>=w:
                styimg[c_x0:c_x0+h,c_y0:c_y0+w,:] = img
                ###sjdsakdsl

                with open(os.path.join(labels, syt_txt + '.txt'), 'w') as f:
                        context = '{},{},{},{}'.format(c_y0,c_x0,c_y0+w, c_x0+h)#x1,y1,x2,y2
                        yolov5_label ='{} {:.3f} {:.3f} {:.3f} {:.3f}'.format(13,(c_y0+w/2)/w1,(c_x0+h/2)/h1,w/w1,h/h1)                       #class x y w h
                        # f.write(context)
                        f.write(yolov5_label)
        elif h1-c_x0>=h and w1-c_y0<=w:
                styimg[c_x0:c_x0 + h, c_y0:, :] = img[:,:w1-c_y0,:]

                with open(os.path.join(labels, syt_txt + '.txt'), 'w') as f:
                        context = '{},{},{},{}'.format(c_y0, c_x0, w1, c_x0 + h)#x1,y1,x2,y2
                        yolov5_label ='{} {:.3f} {:.3f} {:.3f} {:.3f}'.format(13,((w1+c_y0)/2)/w1,(c_x0+h/2)/h1,(w1-c_y0)/w1,h/h1)                       #class x y w h
                        # f.write(context)
                        f.write(yolov5_label)
        elif h1-c_x0<=h and w1-c_y0>=w:
                styimg[c_x0:, c_y0:c_y0+w, :] = img[:h1-c_x0, :, :]

                with open(os.path.join(labels, syt_txt + '.txt'), 'w') as f:
                        context = '{},{},{},{}'.format(c_y0, c_x0, c_y0 + w, h1)#x1,y1,x2,y2
                        yolov5_label ='{} {:.3f} {:.3f} {:.3f} {:.3f}'.format(13,(c_y0+w/2)/w1,((h1+c_x0)/2)/h1,w/w1,(h1-c_x0)/h1)                       #class x y w h
                        # f.write(context)
                        f.write(yolov5_label)
        else:
                styimg[c_x0:, c_y0:, :] = img[:h1 - c_x0, :w1-c_y0, :]

                with open(os.path.join(labels, syt_txt + '.txt'), 'w') as f:
                        context = '{},{},{},{}'.format(c_y0, c_x0, w1,  h1)  #x1,y1,x2,y2
                        yolov5_label ='{} {:.3f} {:.3f} {:.3f} {:.3f}'.format(13,((w1+c_y0)/2)/w1,((h1+c_x0)/2)/h1,(w1-c_y0)/w1,(h1-c_x0)/h1)           #class x y w h
                        # f.write(context)
                        f.write(yolov5_label)
        cv2.imwrite(mix_pic + '/' + img_name, styimg)

# for txt_name in os.listdir(labels):
#         txt_path = os.path.join(labels,txt_name)
#         with open(txt_path,'rw') as f:
#                 text = f.readline()
