import os
import cv2
data_dir = 'D:/data_23/data_23'
save_data_dir = 'D:/temple_deepfashion2/'
i=0
# for file in os.listdir(data_dir):
#     file_path = os.path.join(data_dir,file)
#     if os.path.isdir(file_path):
#         for pic_name in os.listdir(file_path):
#             pic_path = os.path.join(file_path,pic_name)
#             img = cv2.imread(pic_path)
#             cv2.imwrite(save_data_dir+'images/train2017/sal_{}'.format(i)+pic_name,img)
#             i+=1


#########分割 训练集 和 验证集 图片
img_dir =os.path.join(save_data_dir,'images/')
# img_file = os.listdir(img_dir+'train2017')
# img_file.sort()
# print(len(img_file))
# for i,img_name in enumerate(img_file):
#     img_path = os.path.join(img_dir+'train2017',img_name)
#     if i>10000:
#
#         img = cv2.imread(img_path)
#         print(img_dir+'val2017/'+img_name)
#         cv2.imwrite(img_dir+'val2017/'+img_name,img)

########       给图片加标签信息

labels_path = os.path.join(save_data_dir,'labels/')
for folder in os.listdir(img_dir):
    folder_path = os.path.join(img_dir,folder)
    # os.mkdir(os.path.join(labels_path,folder))
    for img_name in os.listdir(folder_path):
        name = img_name.split('.')[0]
        with open(os.path.join(labels_path,folder,name+'.txt'),'w') as f:
            context = '14 0.5 0.5 1 1'
            f.write(context)


