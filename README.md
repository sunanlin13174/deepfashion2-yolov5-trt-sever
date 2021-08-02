# deepfashion2-yolov5-trt-sever
![image](https://github.com/sunanlin13174/deepfashion2-yolov5-trt-sever/edit/main/succeed_find_the_obj.png)  
##这是一个服装检索的项目，使用的数据集是大型服装混合数据集deepfashion2，基于yolov5的backbone的特征图进行衣服图案相似性检索。<br>
<br>项目阶段：<br>
<br>一、训练yolov5并提取特征编码<br>
<br>二、服务器端flask部署<br>
<br>三、本地tensorrt加速<br>
<br>四、tensorrt加速后的模型重新部署到服务器端<br>

<br>第一阶段：<br>
1.下载标注的服装数据集deepfashion2，并用deepfashion2coco.py将一定数量的数据集转为coco格式，修改参数运行两次分别会生成traion.json和val.json，然后运行extrect_some_example.py提取图片到指定的文件夹，图片数量与deepfashion2coco.py文件中一致。<br>
<br>2. 修改example.py中的路径信息并运行，将coco的json文件转为yolov5需要的txt格式。<br>
<br>3.修改train.py中的路径信息，参数设置，即可训练，训练后的模型保存在runs/train/weights下面。<br>
<br>4.运行detect.py可完成推理，修改相应参数。<br>
<br>5.运行search_demo.py即可对图像进行检索，将要检索的一张图像放入新建的demo文件夹中，然后修改search_demo.py中的部分参数即可根据相似性进行图片检索。
使用的yolov5自身的特征向量，并未引入新的网络。输出结果会保存在runs/detect/exp...下面，查询准确率90%+<br>

<br>第二阶段：<br>
<br>一个Flask服务器端，部署的代码在model_sever文件夹。<br>
<br>修改search_rank_5.py的参数，然后python运行即可，相信你看得懂。<br>
<br>返回为gallery文件夹里面与query图片最相似的5张图片，不懂就问，欢迎star和留言。<br>


<br>第三阶段：<br>

<br>YOLOV5目前无法onnx-》engine，部分操作ops官方不支持，所以，可以使用https://github.com/wang-xinyu/tensorrtx.git的方法，<br>
<br>他使用了trt的API搭建了网络，生成了engine，调用时可用python和C++。不过编译前记得修改yololayer.h和.cu的参数。<br>另外，我遇到了undefined reference to `nvinfer1::YoloPluginCreator::YoloPluginCreator()' 这样的坑，谁会的话可以帮帮我。<br>

<br>第四阶段：<br>
待写。。。


