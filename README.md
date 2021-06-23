# deepfashion2-yolov5-trt-sever
##这是一个服装检索的项目，包括以下步骤：<br>
1.下载标注的服装数据集deepfashion2，并用deepfashion2coco.py将一定数量的数据集转为coco格式，分别会生成traion.json和val.json，然后运行extrect_some_example.py提取图片到指定的文件夹，图片数量与deepfashion2coco.py文件中一致。\<br>
<br>2.修改train.py中的路径信息，参数设置，即可训练，训练后的模型保存在runs/train/weights下面。/<br>
<br>3.运行detect.py可完成推理，修改相应参数。/<br>
<br>4.运行search_demo.py即可对图像进行检索，将要检索的一张图像放入新建的demo文件夹中，然后修改search_demo.py中的部分参数即可根据相似性进行图片检索。
使用的yolov5自身的特征向量，并未引入新的网络。输出结果会保存在runs/detect/exp...下面\<br>



