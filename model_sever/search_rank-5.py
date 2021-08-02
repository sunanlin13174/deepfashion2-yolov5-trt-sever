import argparse
import time
from pathlib import Path
import math
import cv2
import heapq
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from PIL import Image
from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path, save_one_box
from utils.plots import colors, plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized
from torchvision.transforms import transforms

from flask import request,Flask

@torch.no_grad()
def detect(opt,query_imgpath,gallery_folder):
    build_transform = transforms.Compose([
        transforms.ToTensor()
    ])
    source_gallery,source_query,weights, view_img = gallery_folder,query_imgpath, opt.weights, opt.view_img
    webcam = source_gallery.isnumeric() or source_gallery.endswith('.txt') or source_gallery.lower().startswith(
        ('rtsp://', 'rtmp://', 'http://', 'https://'))

    # Directories

    # Initialize
    set_logging()
    device = select_device(opt.device)
    half = opt.half and device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    if half:
        model.half()  # to FP16

    # Set Dataloader
    vid_path, vid_writer = None, None
    if webcam:
        view_img = check_imshow()
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source_gallery, stride=stride)

    else:
        dataset = LoadImages(source_gallery,  stride=stride)

    # Run inference
    if device.type != 'cpu':
        model(torch.zeros(1, 3, 416, 416).to(device).type_as(next(model.parameters())))  # run once
    model.eval()
    t0 = time.time()
    ############################ 运行yolov5检测带查询图片中的目标并提取其特征#######################

    img = cv2.imread(query_imgpath)           #读取图片HWC
    img = cv2.resize(img,(640,640))           # 大尺寸
    im0 = img.copy()
    img = img.transpose(2,0,1)                #变为CHW
    img = torch.from_numpy(img).to(device)
    img = img.half() if half else img.float()  # uint8 to fp16/32
    img /= 255.0  # 0 - 255 to 0.0 - 1.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)
        print(img.shape)
    # Inference
    t1 = time_synchronized()
    pred,vetors = model(img, augment=opt.augment)           ################## pred包含两部分，我们只要【0】,可print pred【0】
    # 使用 NMS
    pred = non_max_suppression(pred[0], opt.conf_thres, opt.iou_thres, opt.classes, opt.agnostic_nms,
                               max_det=opt.max_det)
    t2 = time_synchronized()
    for i, det in enumerate(pred):  # detections per image
        if det is not None and len(det) > 0:
            # Rescale boxes from img_size to im0 size
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
            # Write results
            query_loc=[]
            query_img=[]
            areas=[]
            for *xyxy, conf, cls in reversed(det):
                    xmin = int(xyxy[0])
                    ymin = int(xyxy[1])
                    xmax = int(xyxy[2])
                    ymax = int(xyxy[3])
                    w = xmax-xmin
                    h = ymax-ymin
                    areas.append(w*h)
                    query_loc.append((xmin, ymin, xmax, ymax))
            index = np.argmax(areas)         #一张图片可能有多个目标，这里将bbox面积最大的裁剪下来，提取特征。
            xmin,ymin,xmax,ymax = query_loc[index]
            crop_img = im0[int((5/4)*ymin):int((3/4)*ymax), int((5/4)*xmin):int((3/4)*xmax)]  # 这里将检测的bbox进一步中心裁剪，滤去多余的背景，以提高查询精度。
            # crop_img = im0[ymin:ymax,xmin:xmax]
            crop_img = cv2.resize(crop_img, (416, 416), interpolation=cv2.INTER_LINEAR)
            cv2.imshow('dsd.jpg',crop_img)        # 调试使用，本地可视化输入目标
            cv2.waitKey(30)
            crop_img = Image.fromarray(cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB))
            crop_img = build_transform(crop_img).unsqueeze(0)  # torch.Size([1, 3, 416,416])
            query_img.append(crop_img)      #
            if query_img:
                query_img = torch.cat(query_img, dim=0)         #维度拼接，可删除
                print(query_img.shape)    # 1,3,416,416
                query_img = query_img.to(device)
                out , queryfeats  = model(query_img, augment=opt.augment)   ##wrong queryfeats is list
                print(queryfeats[0].shape)
                queryfeats = torch.nn.functional.normalize(queryfeats[2], dim=1, p=2)
############################ 遍历gallery文件夹下的图片，yolov5检测+特征提取#######################
    whole_img_distance = []                            #存储查询图片和gallery下的每一张图片特征向量的距离，为了后面挑出最小的x张图片。
    whole_det_imgpath =[]                                 #按照whole_img_distance[]的存储顺序依次存储 gallery下的图片，按照返回的索引，显示挑出的图片，以及返回它们的路径
    look_img=[]
    for path, img, im0s, vid_cap in dataset:           #以dataload的方式加载 gallery数据集
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
        # Inference
        t3 = time_synchronized()
        pred,vetors = model(img, augment=opt.augment)            ###############
        # Apply NMS
        pred = non_max_suppression(pred[0], opt.conf_thres, opt.iou_thres, opt.classes, opt.agnostic_nms,
                                   max_det=opt.max_det)
        for i, det in enumerate(pred):  # detections per image
            if webcam:  # batch_size >= 1
                p, s, im0, frame = path[i], f'{i}: ', im0s[i].copy(), dataset.count
            else:
                p, s, im0, frame = path, '', im0s.copy(), getattr(dataset, 'frame', 0)
            p = Path(p)  # to Path

            if det is not None and len(det) > 0:
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
                # Write results
                gallery_loc=[]
                gallery_img=[]
                for *xyxy, conf, cls in reversed(det):
                        xmin = int(xyxy[0])
                        ymin = int(xyxy[1])
                        xmax = int(xyxy[2])
                        ymax = int(xyxy[3])
                        gallery_loc.append((xmin, ymin, xmax, ymax))
                        # crop_img = im0[int((5 / 4) * ymin):int((3 / 4) * ymax), int((5 / 4) * xmin):int((3 / 4) * xmax)]
                        crop_img = im0[ymin:ymax,xmin:xmax]  # HWC (H,W,3)
                        crop_img=cv2.resize(crop_img,(224,224))##########图像细节，可视化输入yolov5的crop_img与gallery——img可知，bbox裁剪出的图像像素太低，而原图设计的像素太高，因此先将原图设计的图片进行
                        # #resize降低分辨率，然后在扩充到统一尺寸，使其模糊，与bbox裁剪的图像保持一致处理，效果不错。
                        # # crop_img = im0[int((5/4)*ymin):int((3/4)*ymax), int((5/4)*xmin):int((3/4)*xmax)]  # HWC (602, 233, 3)
                        # crop_img = cv2.medianBlur(crop_img,3)
                        crop_img = cv2.resize(crop_img, (416, 416), interpolation=cv2.INTER_LINEAR)
                        crop_img = Image.fromarray(cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB))  # PIL: (233, 602)
                        crop_img = build_transform(crop_img).unsqueeze(0)  # torch.Size([1, 3, 256, 128])
                        gallery_img.append(crop_img)
                if gallery_img:
                    gallery_img = torch.cat(gallery_img, dim=0)
                    gallery_img = gallery_img.to(device)
                    out, galleryfeats  = model(gallery_img, augment=opt.augment)
                    # print(galleryfeats.shape)
                    galleryfeats = torch.nn.functional.normalize(galleryfeats[2], dim=1, p=2)
                    #########  for循环内，即查询图片与每张gallery_img计算特征距离，计算方式为欧氏距离，可优化。
                    queryfeats = queryfeats.view(-1)
                    galleryfeats = galleryfeats.view(-1)
                    # print(feat.shape)
                    # distmat = torch.dot(feat,gall_feat)/(torch.linalg.norm(feat)*torch.linalg.norm(gall_feat))
                    # distmat = cosine_similarity(feat,gall_feat).reshape(1,-1)     #1024x1024
                    distmat = 1 - torch.cosine_similarity(galleryfeats, queryfeats, dim=0)

                    # distmat = torch.sqrt(((queryfeats - galleryfeats) ** 2).view(-1,1)).mean(0)    ##1024x20x20.view().mean()
                    distmat = distmat.cpu().numpy()  # <class 'tuple'>: (3, 12)
                    index = distmat.argmin()         # 挑出gallery_img的多个bbox与查询图片的特征距离最小的bbox，作为此gallery_img的代表。
                    # print('距离：',distmat[index])
                    if distmat[index]<0.05:          # 筛选，若最小距离小于0.05，则去掉，0.05为超参数，根据距离计算方式调整。
                        whole_img_distance.append(distmat[index])     #存下 该图片与查询图片的距离
                        plot_one_box(gallery_loc[index], im0, label='find_meprint_logo!') #对原图画框框，可选
                        whole_det_imgpath.append(path)                    # 存下满足条件的图片路径
                        look_img.append(im0)

            t4 = time_synchronized()
            # print('one box spend {}s'.format(t4-t3))
    print('一共查询的图片数，每张有一个相似目标：', len(whole_img_distance))
    min_value = heapq.nsmallest(10,whole_img_distance)
    print(min_value)
    min_dist = list(map(whole_img_distance.index,min_value))
    print('最相似图片的列表索引位置：', min_dist)

    #返回最相似的图片路径，json文件。
    best_imgs_path = []
    for i in min_dist:
        best_img = look_img[i]
        cv2.imwrite('{}.jpg'.format(i),best_img)
        best_imgpath = whole_det_imgpath[i]
        best_imgs_path.append(best_imgpath)
    result = {'find':best_imgs_path}
    return result
app = Flask(__name__)
@app.route('/search',methods=['POST'])
def search(gallery_folder='E:/search_gallery'):
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='E:/best (1).pt', help='model.pt path(s)')
    parser.add_argument('--gallery', type=str, default='E:/search_gallery',
                        help='source_gallery')  # file/folder, 0 for webcam
    parser.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum number of detections per image')
    parser.add_argument('--device', default='0', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-crop', default='False', action='store_true', help='save cropped prediction boxes')
    parser.add_argument('--nosave', default='True', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    # parser.add_argument('--project', default='runs/detect', help='save results to project/name')
    # parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--line-thickness', default=3, type=int, help='bounding box thickness (pixels)')
    parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
    parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')
    parser.add_argument('--half', type=bool, default=False, help='use FP16 half-precision inference')
    opt = parser.parse_args()

    jsons_input = request.get_json()
    print(jsons_input)
    query_imgpath = jsons_input["query_img"]
    print(query_imgpath)
    if jsons_input['gallery_folder']:          # 如果传入的json文件为{ "query_img":"E:/search_query/mepaint_2.png","gallery_folder":""}，则search函数的默认参数会执行。
        gallery_folder = jsons_input['gallery_folder']
        print('__________________________')
    out = detect(opt,query_imgpath,gallery_folder)
    return out

if __name__ == '__main__':
     app.run('127.0.0.2',port=5000)