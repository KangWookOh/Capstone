"""
20220725 : batch_iou, nms 함수 추가 (김진수)
20220726 : yolo2voc, voc2yolo array([])가 입력될 때 오류 수정 (김진수)
20220826 : validsplit, testsplit 함수 추가 (김혜빈)
20220929 : softnms 추가 (김진수)
20220930 : yolo2coco 추가 (김진수)
XXXXXXXX : crop_detection 추가(김진수)
20221006 : selection 추가 (김진수)
"""

import torch
from models.common import DetectMultiBackend
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadStreams
from utils.general import (LOGGER, check_file, check_img_size, check_imshow, check_requirements, colorstr,
                           increment_path, non_max_suppression, print_args, scale_boxes, strip_optimizer, xyxy2xywh)
from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import select_device, time_sync
from utils.augmentations import letterbox
import glob
import copy
import cv2
import numpy as np
from PIL import Image, ImageDraw
import random
import shutil

def imgload(Imgpath):
    img = cv2.imread(Imgpath)
    if img is None:
        img = Image.open(Imgpath)
        img = cv2.cvtColor(np.uint8(img), cv2.COLOR_BGR2RGB)
    return img

def imshow(img):
    cv2.imshow('image', img)
    cv2.waitKey() 
    cv2.destroyAllWindows()
    
def macshow(img):
    _img = copy.deepcopy(img)
    p_img = Image.fromarray(_img)
    p_img.show()

def imread(filename, flags=cv2.IMREAD_COLOR, dtype=np.uint8):
    try:
        n = np.fromfile(filename, dtype)
        img = cv2.imdecode(n, flags)
        return img
    except Exception as e:
        print(e)
        return None
    
def yolo2voc(Img, Txt, mode='gt'):
    _Txt = copy.deepcopy(Txt)
    m, n, k = Img.shape
    if len(_Txt) > 0:
        if mode == 'gt':
            v_x, v_y = (_Txt[:,1] - _Txt[:,3]/2) * n + 1, (_Txt[:,2] - _Txt[:,4]/2) * m + 1
            v_w, v_h = (_Txt[:,1] + _Txt[:,3]/2) * n + 1, (_Txt[:,2] + _Txt[:,4]/2) * m + 1
               
            _Txt[:,1], _Txt[:,2], _Txt[:,3], _Txt[:,4] = v_x, v_y, v_w, v_h
            _Txt = _Txt.astype(int)
            
        elif mode == 'det':
            v_x, v_y = (_Txt[:,2] - _Txt[:,4]/2) * n + 1, (_Txt[:,3] - _Txt[:,5]/2) * m + 1
            v_w, v_h = (_Txt[:,2] + _Txt[:,4]/2) * n + 1, (_Txt[:,3] + _Txt[:,5]/2) * m + 1
            
            _Txt[:,2], _Txt[:,3], _Txt[:,4], _Txt[:,5] = v_x, v_y, v_w, v_h
    return _Txt

def voc2yolo(Img, Txt, mode='gt'):
    _Txt = copy.deepcopy(Txt)
    m, n, k = Img.shape
     
    if mode == 'gt':
        y_x, y_y = ((_Txt[:,3] + _Txt[:,1]) / 2 - 1) / n, ((_Txt[:,4] + _Txt[:,2]) / 2 - 1) / m
        y_w, y_h = (_Txt[:,3] - _Txt[:,1] - 2) / n, (_Txt[:,4] - _Txt[:,2] - 2) / m
        
        _Txt[:,1], _Txt[:,2], _Txt[:,3], _Txt[:,4] = y_x, y_y, y_w, y_h
        # _Txt = _Txt.astype(int)
        
    elif mode == 'det':
        y_x, y_y = ((_Txt[:,4] + _Txt[:,2]) / 2 - 1) / n, ((_Txt[:,5] + _Txt[:,3]) / 2 - 1) / m
        y_w, y_h = (_Txt[:,4] - _Txt[:,2] - 2) / n, (_Txt[:,5] - _Txt[:,3] - 2) / m
        
        _Txt[:,2], _Txt[:,3], _Txt[:,4], _Txt[:,5] = y_x, y_y, y_w, y_h       
        
    return _Txt


def yolo2coco(Img, Txt, mode='gt'):
    _Txt = copy.deepcopy(Txt)
    m, n, k = Img.shape
    if len(_Txt) > 0:
        if mode == 'gt':
            v_x, v_y = (_Txt[:,1] - _Txt[:,3]/2) * n, (_Txt[:,2] - _Txt[:,4]/2) * m
            v_w, v_h = Txt[:,3] * n, _Txt[:,4] * m
            _Txt[:,1], _Txt[:,2], _Txt[:,3], _Txt[:,4] = v_x, v_y, v_w, v_h
            # _Txt = _Txt.astype(int)
            
        elif mode == 'det':
            v_x, v_y = (_Txt[:,2] - _Txt[:,4]/2) * n, (_Txt[:,3] - _Txt[:,5]/2) * m
            v_w, v_h = Txt[:,4] * n, _Txt[:,5] * m
            _Txt[:,2], _Txt[:,3], _Txt[:,4], _Txt[:,5] = v_x, v_y, v_w, v_h
            
    return _Txt


def setcolor(mode='basic'):
    if mode == 'basic':
        Colors = [(0, 0, 255), (255, 0, 0), (0, 255, 0), (255, 255, 0)]
    
    elif mode == 'random':
        
        # Colors = [[random.randint(0,255) for j in range(3)] for i in range(len(Classes))]
        Colors = [[random.randint(0,255) for j in range(3)] for i in range(200)]
        for k in range(len(Colors)):
            Colors[k] = tuple(Colors[k])
    return Colors


def preprocessing(Img, Model, Device):
    _Img = copy.deepcopy(Img)
    _Img = letterbox(_Img, 640, 32, True)[0]
    _Img = _Img.transpose((2, 0, 1))[::-1]
    _Img = np.ascontiguousarray(_Img)
    im = torch.from_numpy(_Img).to(Device)
    im = im.half() if Model.fp16 else im.float()  # uint8 to fp16/32
    im /= 255  # 0 - 255 to 0.0 - 1.0
    if len(im.shape) == 3:
        im = im[None]  # expand for batch dim
    return im

def v5_load(Path, mode='gpu'):
    weights = Path
    if mode == 'gpu':
        device = torch.device('cuda:0')
    elif mode == 'cpu':
        device = torch.device('cpu')        
    model = DetectMultiBackend(weights, device=device, dnn=False, data=None, fp16=False)
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size((640, 640), s=stride)
    model.warmup(imgsz=(1 if pt else bs, 3, *imgsz))
    return model, names, device  


def v5_det(_Img, Img, Model):

    pred = Model(_Img, augment=False, visualize=False)
    conf_thres, iou_thres, classes, agnostic_nms, max_det, = 0.25, 0.45, None, False, 1000
    pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
    
    data = []
    for det in pred:
        if len(det):
            det[:,:4] = scale_boxes(_Img.shape[2:], det[:,:4], Img.shape).round()
            for d in det:
                data.append([int(d[5]), round(float(d[4]), 2), int(d[0]), int(d[1]), int(d[2]), int(d[3])])

                
    data = np.array(data, dtype=np.float64)
    if len(data) > 0:
        data = voc2yolo(Img, data, mode='det')
    
    return data


def v4_load(Path, Cfg, Weights, Names, mode='cpu'):
    if mode == 'cpu':
        Net = cv2.dnn.readNet(Path + '/' + Cfg + '.cfg', Path + '/' + Weights +  '.weights')
        with open(Path + '/' + Names + '.names', 'r') as f:
            Classes = [line.strip() for line in f.readlines()]
        layer_names = Net.getLayerNames()
        Output_layers = [layer_names[i[0] - 1] for i in Net.getUnconnectedOutLayers()]
    return Net, Output_layers, Classes



def v4_det(Net, Output_layers, Classes, Img, mode='cpu'):
    if mode == 'cpu':
        _Img = copy.deepcopy(Img)    
        if type(_Img) == np.ndarray : h, w, c = _Img.shape
        elif type(_Img) == PIL.JpegImagePlugin.JpegImageFile:  
            _Img = cv2.cvtColor(np.uint8(_Img), cv2.COLOR_BGR2RGB)
            h, w, c = _Img.shape
        
            
        blob = cv2.dnn.blobFromImage(_Img, 0.00392, (608, 608), (0, 0, 0), True, crop=False)
        Net.setInput(blob)
        outs = Net.forward(Output_layers)
        
        class_ids = []
        confidences = []
        boxes = []
        
        for out in outs:
            for detection in out:
                
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                
                if confidence > 0.5:
                    center_x = int(detection[0] * w)
                    center_y = int(detection[1] * h)
                    dw = int(detection[2] * w)
                    dh = int(detection[3] * h)
                    
                    x = int(center_x - dw / 2)
                    y = int(center_y - dh / 2)
                    boxes.append([x, y, dw, dh])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)
                    
        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.45, 0.4)
        
        data = []
        for b in range(len(boxes)):
            if b in indexes:
                xx, yy, ww, hh = boxes[b]
                label = str(Classes[class_ids[b]])
                score = round(confidences[b], 2)
                
                for c in range(len(Classes)):
                    if label == Classes[c]:
                        data.append([c, score, xx, yy, xx+ww, yy+hh])
                        # cv2.rectangle(_Img, (xx, yy), (xx + ww, yy + hh), (255, 0, 0), 2)
                        
        data = np.array(data, dtype=np.float64)
        if len(data) > 0:
            data = voc2yolo(_Img, data, mode='det')
    
    return data

def selection(Txt, Classes, mode='gt'):
    _Txt = copy.deepcopy(Txt)
    sel = []
    for lab in _Txt:
        if lab[0] in Classes:
            sel.append(lab)
    sel = np.array(sel)
    
    return sel

def yolodraw(Img, Txt, Colors, Classes, mode='gt'):
    _Img = copy.deepcopy(Img)
    _Txt = copy.deepcopy(Txt)
    
    if mode == 'gt':
        _Txt = yolo2voc(Img, _Txt, mode='gt')
        for lab in _Txt:            
            if lab[0] in Classes:            
                color = Colors[int(lab[0])]
                _Img = cv2.rectangle(_Img, (lab[1], lab[2]), (lab[3], lab[4]), color, 10)
                
    if mode == 'det':
        _Txt = yolo2voc(Img, _Txt, mode='det')
        for lab in _Txt:
            if lab[0] in Classes:  
                color = Colors[int(lab[0])]
                _Img = cv2.rectangle(_Img, (int(lab[2]), int(lab[3])), (int(lab[4]), int(lab[5])), color, 2)
    return _Img

def batch_iou(boxes, box):
    lr = np.maximum(np.minimum(boxes[:,0]+0.5*boxes[:,2], box[0]+0.5*box[2]) - np.maximum(boxes[:,0]-0.5*boxes[:,2], box[0]-0.5*box[2]), 0)
    tb = np.maximum(np.minimum(boxes[:,1]+0.5*boxes[:,3], box[1]+0.5*box[3]) - np.maximum(boxes[:,1]-0.5*boxes[:,3], box[1]-0.5*box[3]), 0)
    inter = lr * tb
    union = boxes[:,2]*boxes[:,3] + box[2]*box[3] - inter
    return inter/union

def nms(Boxes, Probs, Thresh):
    order = Probs.argsort()[::-1]
    keep = [True]*len(order)
    
    for i in range(len(order)-1):
        ovps = batch_iou(Boxes[order[i+1:]], Boxes[order[i]])
        for j, ov in enumerate(ovps):
            if ov > Thresh:
                keep[order[j+i+1]] = False
                
    return keep

def validsplit(Origin, Train, Valid, train_ratio, mode='basic'):
    dir_img = glob.glob(Origin + ('*.jpeg'))
    dir_txt = glob.glob(Origin + ('*.txt'))
    
    if mode == 'basic':
        train_num = int(train_ratio * len(dir_img))

        train_imgs, train_txts = dir_img[0:train_num - 1], dir_txt[0:train_num - 1]
        valid_imgs, valid_txts = dir_img[train_num : len(dir_img)], dir_txt[train_num : len(dir_img)]
        
        for i in range(len(train_imgs)):
            shutil.copy2(train_imgs[i], Train  + ('images'))
            shutil.copy2(train_txts[i], Train  + ('labels'))
            
        for j in range(len(valid_imgs)):
            shutil.copy2(valid_imgs[j], Valid  + ('images'))
            shutil.copy2(valid_txts[j], Valid  + ('labels'))
        
    if mode == 'random':
        tmp = [[x,y] for x, y in zip(dir_img, dir_txt)]
        random.shuffle(tmp)

        dir_img = [n[0] for n in tmp]
        dir_txt = [n[1] for n in tmp]
        
        train_num = int(train_ratio * len(dir_img))

        train_imgs, train_txts = dir_img[0:train_num - 1], dir_txt[0:train_num - 1]
        valid_imgs, valid_txts = dir_img[train_num : len(dir_img)], dir_txt[train_num : len(dir_img)]


        for i in range(len(train_imgs)):
            shutil.copy2(train_imgs[i], Train  + ('images'))
            shutil.copy2(train_txts[i], Train  + ('labels'))
            
        for j in range(len(valid_imgs)):
            shutil.copy2(valid_imgs[j], Valid  + ('images'))
            shutil.copy2(valid_txts[j], Valid  + ('labels'))
            
def testsplit(Origin, Train, Valid, Test, train_ratio, valid_ratio, mode='basic'):
    dir_img = glob.glob(Origin + ('*.jpeg'))
    dir_txt = glob.glob(Origin + ('*.txt'))
    
    if mode == 'basic':
        train_num = int(train_ratio * len(dir_img))
        valid_num = int(valid_ratio * len(dir_img))
        
        train_imgs, train_txts = dir_img[0:train_num - 1], dir_txt[0:train_num - 1]
        valid_imgs, valid_txts = dir_img[train_num : (train_num + valid_num)-1], dir_txt[train_num : (train_num + valid_num)-1]
        test_imgs, test_txts = dir_img[train_num + valid_num : len(dir_img)], dir_txt[train_num + valid_num : len(dir_img)]
        
        for i in range(len(train_imgs)):
            shutil.copy2(train_imgs[i], Train  + ('images'))
            shutil.copy2(train_txts[i], Train  + ('labels'))
            
        for j in range(len(valid_imgs)):
            shutil.copy2(valid_imgs[j], Valid  + ('images'))
            shutil.copy2(valid_txts[j], Valid  + ('labels'))
            
        for k in range(len(test_imgs)):
            shutil.copy2(test_imgs[k], Test  + ('images'))
            shutil.copy2(test_txts[k], Test  + ('labels'))
        
    if mode == 'random':
        tmp = [[x,y] for x, y in zip(dir_img, dir_txt)]
        random.shuffle(tmp)

        dir_img = [n[0] for n in tmp]
        dir_txt = [n[1] for n in tmp]
        
        train_num = int(train_ratio * len(dir_img))
        valid_num = int(valid_ratio * len(dir_img))
        
        train_imgs, train_txts = dir_img[0:train_num - 1], dir_txt[0:train_num - 1]
        valid_imgs, valid_txts = dir_img[train_num : (train_num + valid_num)-1], dir_txt[train_num : (train_num + valid_num)-1]
        test_imgs, test_txts = dir_img[train_num + valid_num : len(dir_img)], dir_txt[train_num + valid_num : len(dir_img)]
        
        for i in range(len(train_imgs)):
            shutil.copy2(train_imgs[i], Train  + ('images'))
            shutil.copy2(train_txts[i], Train  + ('labels'))
            
        for j in range(len(valid_imgs)):
            shutil.copy2(valid_imgs[j], Valid  + ('images'))
            shutil.copy2(valid_txts[j], Valid  + ('labels'))
            
        for k in range(len(test_imgs)):
            shutil.copy2(test_imgs[k], Test  + ('images'))
            shutil.copy2(test_txts[k], Test  + ('labels'))



def det2tensor(Img, Txt):
    _Txt = copy.deepcopy(Txt)
    _Txt = yolo2voc(Img, _Txt, mode='det')
    
    x1, y1, x2, y2 = copy.deepcopy(_Txt[:,2]), copy.deepcopy(_Txt[:,3]), copy.deepcopy(_Txt[:,4]), copy.deepcopy(_Txt[:,5])    
    _Txt[:,2], _Txt[:,3], _Txt[:,4], _Txt[:,5] = y1, x1, y2, x2
    
    bbox, score = [], []
    for i in range(len(_Txt)):
        val = _Txt[i]
        box = list(val[2:6])
        box = [int(j) for j in box]
        bbox.append(box)
        score.append(val[1])
        
    return bbox, score
    
            
def soft_nms_pytorch(dets, box_scores, sigma=0.5, thresh=0.7, cuda=0):
    """
    Build a pytorch implement of Soft NMS algorithm.
    # Augments
        dets:        boxes coordinate tensor (format:[y1, x1, y2, x2])
        box_scores:  box score tensors
        sigma:       variance of Gaussian function
        thresh:      score thresh
        cuda:        CUDA flag
    # Return
        the index of the selected boxes
    """

    # Indexes concatenate boxes with the last column
    N = dets.shape[0]
    if cuda:
        indexes = torch.arange(0, N, dtype=torch.float).cuda().view(N, 1)
    else:
        indexes = torch.arange(0, N, dtype=torch.float).view(N, 1)
    dets = torch.cat((dets, indexes), dim=1)

    # The order of boxes coordinate is [y1,x1,y2,x2]
    y1 = dets[:, 0]
    x1 = dets[:, 1]
    y2 = dets[:, 2]
    x2 = dets[:, 3]
    scores = box_scores
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)

    for i in range(N):
        # intermediate parameters for later parameters exchange
        tscore = scores[i].clone()
        pos = i + 1

        if i != N - 1:
            maxscore, maxpos = torch.max(scores[pos:], dim=0)
            if tscore < maxscore:
                dets[i], dets[maxpos.item() + i + 1] = dets[maxpos.item() + i + 1].clone(), dets[i].clone()
                scores[i], scores[maxpos.item() + i + 1] = scores[maxpos.item() + i + 1].clone(), scores[i].clone()
                areas[i], areas[maxpos + i + 1] = areas[maxpos + i + 1].clone(), areas[i].clone()

        # IoU calculate
        yy1 = np.maximum(dets[i, 0].to("cpu").numpy(), dets[pos:, 0].to("cpu").numpy())
        xx1 = np.maximum(dets[i, 1].to("cpu").numpy(), dets[pos:, 1].to("cpu").numpy())
        yy2 = np.minimum(dets[i, 2].to("cpu").numpy(), dets[pos:, 2].to("cpu").numpy())
        xx2 = np.minimum(dets[i, 3].to("cpu").numpy(), dets[pos:, 3].to("cpu").numpy())
        
        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = torch.tensor(w * h).cuda() if cuda else torch.tensor(w * h)
        ovr = torch.div(inter, (areas[i] + areas[pos:] - inter))

        # Gaussian decay
        weight = torch.exp(-(ovr * ovr) / sigma)
        scores[pos:] = weight * scores[pos:]

    # select the boxes and keep the corresponding indexes
    keep = dets[:, 4][scores > thresh].int()

    return keep
    