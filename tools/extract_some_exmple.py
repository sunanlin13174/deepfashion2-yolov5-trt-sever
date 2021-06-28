import os
import shutil
img_path='/home/meprint/sunanlin_folder/train/image'
save_img_path = '/home/meprint/sunanlin_folder/deepfashion2/images/train2017'
label_txtpath='/home/meprint/sunanlin_folder/deepfashion2/labels/train2017'
files= os.listdir(img_path)
files.sort()
for index,txt_name in files:
    if index<=1999:
        shutil.copy(os.path.join(img_path,txt_name),os.path.join(save_img_path,txt_name))



