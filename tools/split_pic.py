import os
import cv2
import numpy as np
import numpy as py
pic_dir = 'D:/meprint_pic'
query = 'D:/search_query/'
gallery = 'D:/search_gallery/'
# if os.path.isdir(query):
#     os.remove(query)
#     os.mkdir(query)
# if os.path.isdir(gallery):
#     os.remove(gallery)
#     os.mkdir(gallery)
files = os.listdir(pic_dir)
files.sort()
for i,pic_name in enumerate(files):
    pic_path = os.path.join(pic_dir,pic_name)
    img = cv2.imread(pic_path)
    w,h,c = img.shape
    img_zeors = np.ndarray((int(w),int(h/2),c),dtype=np.uint8)
    # print(w,h,c)
    img_zeors[:,:,:] = img[:,:int(h/2),:]
    if pic_name.endswith('.png'):
        cv2.imwrite(gallery+'mepaint_{}.png'.format(i),img[:,int(h/2):,:])
        cv2.imwrite(query+'mepaint_{}.png'.format(i),img_zeors)
    if pic_name.endswith('.jpg'):
        cv2.imwrite(gallery+'mepaint_{}.jpg'.format(i),img[int(w/2):,int(h/2):,:])
        cv2.imwrite(query+'mepaint_{}.jpg'.format(i),img_zeors)
