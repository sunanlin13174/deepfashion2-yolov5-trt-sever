import argparse
import time
from pathlib import Path
import math
import cv2
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

@torch.no_grad()
def detect(opt):
    build_transform = transforms.Compose([
        transforms.ToTensor()
    ])
    query_feats = []
    gallery_feats = []
    source_gallery,source_query,weights, view_img,imgsz = opt.gallery,opt.query, opt.weights, opt.view_img, opt.img_size
    save_img = not opt.nosave and not source_gallery.endswith('.txt')  # save inference images
    webcam = source_gallery.isnumeric() or source_gallery.endswith('.txt') or source_gallery.lower().startswith(
        ('rtsp://', 'rtmp://', 'http://', 'https://'))

    # Directories
    save_dir = increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok)  # increment run
    (save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Initialize
    set_logging()
    device = select_device(opt.device)
    half = opt.half and device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check img_size
    names = model.module.names if hasattr(model, 'module') else model.names  # get class names
    if half:
        model.half()  # to FP16

    # Set Dataloader
    vid_path, vid_writer = None, None
    if webcam:
        view_img = check_imshow()
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source_gallery, img_size=imgsz, stride=stride)
        query_set = LoadStreams(source_query,img_size=imgsz, stride=stride)
    else:
        dataset = LoadImages(source_gallery, img_size=imgsz, stride=stride)
        query_set = LoadImages(source_query, img_size=imgsz, stride=stride)
    # Run inference
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
    model.eval()
    t0 = time.time()
    ############################ extrect vetors of query images#######################
    for path, img, im0s, vid_cap in query_set:
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
        # Inference
        t1 = time_synchronized()
        pred,vetors = model(img, augment=opt.augment)           ###############
        # Apply NMS
        ###pres包含两部分，我们只要【0】
        pred = non_max_suppression(pred[0], opt.conf_thres, opt.iou_thres, opt.classes, opt.agnostic_nms,
                                   max_det=opt.max_det)
        t2 = time_synchronized()
        for i, det in enumerate(pred):  # detections per image
            if webcam:  # batch_size >= 1
                p, s, im0, frame = path[i], f'{i}: ', im0s[i].copy(), dataset.count
            else:
                p, s, im0, frame = path, '', im0s.copy(), getattr(dataset, 'frame', 0)

            p = Path(p)  # to Path

            s += '%gx%g ' % img.shape[2:]  # print string
            imc = im0.copy() if opt.save_crop else im0  # for opt.save_crop
            if det is not None and len(det) > 0:
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
                # Write results
                query_loc=[]
                query_img=[]
                for *xyxy, conf, cls in reversed(det):
                        c = int(cls)
                        label = None if opt.hide_labels else (names[c] if opt.hide_conf else f'{names[c]} {conf:.2f}')
                        plot_one_box(xyxy, im0, label=label, color=colors(c, True), line_thickness=opt.line_thickness)
                        xmin = int(xyxy[0])
                        ymin = int(xyxy[1])
                        xmax = int(xyxy[2])
                        ymax = int(xyxy[3])
                        query_loc.append((xmin, ymin, xmax, ymax))
                        crop_img = im0[ymin:ymax, xmin:xmax]  # HWC (602, 233, 3)
                        crop_img = cv2.resize(crop_img,(640,640),interpolation=cv2.INTER_LINEAR)
                        # cv2.imshow('dsd.jpg',crop_img)
                        # cv2.waitKey(0)
                        crop_img = Image.fromarray(cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB))  # PIL: (233, 602)
                        crop_img = build_transform(crop_img).unsqueeze(0)  # torch.Size([1, 3, 256, 128])
                        query_img.append(crop_img)
                if query_img:
                    query_img = torch.cat(query_img, dim=0)
                    print(query_img.shape)    # 1,3,640,640
                    query_img = query_img.to(device)
                    out , queryfeats  = model(query_img, augment=opt.augment)   ##wrong queryfeats is list
                    print(queryfeats[0].shape)
                    queryfeats = torch.nn.functional.normalize(queryfeats, dim=1, p=2)
                    query_feats.append(queryfeats)
                    print('query pics total have:',len(query_feats))

############################ extrect vetors of gallery images#######################
    for path, img, im0s, vid_cap in dataset:
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
            save_path = str(save_dir / p.name)  # img.jpg
            if det is not None and len(det) > 0:
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
                # Write results
                gallery_loc=[]
                gallery_img=[]
                for *xyxy, conf, cls in reversed(det):
                        c = int(cls)
                        label = None if opt.hide_labels else (names[c] if opt.hide_conf else f'{names[c]} {conf:.2f}')
                        plot_one_box(xyxy, im0, label=label, color=colors(c, True), line_thickness=opt.line_thickness)
                        xmin = int(xyxy[0])
                        ymin = int(xyxy[1])
                        xmax = int(xyxy[2])
                        ymax = int(xyxy[3])
                        gallery_loc.append((xmin, ymin, xmax, ymax))
                        crop_img = im0[ymin:ymax, xmin:xmax]  # HWC (602, 233, 3)
                        crop_img = cv2.resize(crop_img, (640, 640), interpolation=cv2.INTER_LINEAR)
                        crop_img = Image.fromarray(cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB))  # PIL: (233, 602)
                        crop_img = build_transform(crop_img).unsqueeze(0)  # torch.Size([1, 3, 256, 128])
                        gallery_img.append(crop_img)
                if gallery_img:
                    gallery_img = torch.cat(gallery_img, dim=0)
                    gallery_img = gallery_img.to(device)
                    out, galleryfeats  = model(gallery_img, augment=opt.augment)
                    print(galleryfeats.shape)
                    galleryfeats = torch.nn.functional.normalize(galleryfeats, dim=1, p=2)
                    gallery_feats.append(galleryfeats)

                    #########  compute the distmat
                    m, n = query_feats[0].shape[0], gallery_feats[0].shape[0]
                    print(m,n)
                    distmat = np.sqrt(((queryfeats-galleryfeats)**2).mean(axis=-1).mean(axis=-1))
                    print('along 1024 dim sum,so should be [2,1024]',distmat.shape)
                    # distmat = torch.pow(queryfeats, 2).sum(dim=1, keepdim=True).expand(m, n) + \
                    #            torch.pow(galleryfeats, 2).sum(dim=1, keepdim=True).expand(n, m).t()

                    # out=(beta∗M)+(alpha∗mat1@mat2)
                    # qf^2 + gf^2 - 2 * qf@gf.t()
                    # distmat - 2 * qf@gf.t()
                    # distmat: qf^2 + gf^2
                    # qf: torch.Size([2, 2048])
                    # gf: torch.Size([7, 2048])
                    distmat = distmat.sum(axis =-1)/1024 # [2,distance]
                    distmat = distmat.pow(1/4)
                    # print(distmat.shape)

                    distmat = distmat.cpu().numpy()  # <class 'tuple'>: (3, 12)
                    index = distmat.argmin()

                    print(distmat)
                    if distmat[index] < opt.dist_thres:
                        print('距离：%s' % distmat[index])
                        plot_one_box(gallery_loc[index], im0, label='find_meprint_logo!')
                        # cv2.imshow('person search', im0)
                        # cv2.waitKey()

            t4 = time_synchronized()
            print('one box spend {}s'.format(t4-t3))

            # Stream results
            if view_img:
                cv2.imshow(str(p), im0)
                cv2.waitKey(1)  # 1 millisecond

            # Save results (image with detections)
            if save_img:
                if dataset.mode == 'image':
                    cv2.imwrite(save_path, im0)
                else:  # 'video' or 'stream'
                    if vid_path != save_path:  # new video
                        vid_path = save_path
                        if isinstance(vid_writer, cv2.VideoWriter):
                            vid_writer.release()  # release previous video writer
                        if vid_cap:  # video
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:  # stream
                            fps, w, h = 30, im0.shape[1], im0.shape[0]
                            save_path += '.mp4'
                        vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                    vid_writer.write(im0)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='E:/best.pt', help='model.pt path(s)')
    parser.add_argument('--gallery', type=str, default='E:/demo_gallery', help='source_gallery')  # file/folder, 0 for webcam
    parser.add_argument('--query', nargs='+', type=str, default='E:/demo', help='query lists')
    parser.add_argument('--dist_thres', type=int, default=0.25, help='dist thres')
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum number of detections per image')
    parser.add_argument('--device', default='0', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-crop', default='True',action='store_true', help='save cropped prediction boxes')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default='runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--line-thickness', default=3, type=int, help='bounding box thickness (pixels)')
    parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
    parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')
    parser.add_argument('--half', type=bool, default=False, help='use FP16 half-precision inference')
    opt = parser.parse_args()
    print(opt)
    check_requirements(exclude=('tensorboard', 'pycocotools', 'thop'))

    if opt.update:  # update all models (to fix SourceChangeWarning)
        for opt.weights in ['yolov5s.pt', 'yolov5m.pt', 'yolov5l.pt', 'yolov5x.pt']:
            detect(opt=opt)
            strip_optimizer(opt.weights)
    else:
        detect(opt=opt)
