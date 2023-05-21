# -*- coding: utf-8 -*-
"""
Created on Sat Oct 29 15:45:15 2022

@author: khb05
"""

import numpy as np, json, os.path, glob, os, cv2
from info import *
import pandas as pd

# 이미지 경로를 불러옵니다요
imgpaths = glob.glob('/Users/kangwook/Desktop/Captone/댕댕/save_img/*.jpg')
# imgpaths = glob.glob('C:/Users/khb05/바탕 화면/dataset/valid/images/IMG_D_A1_011211.jpg')

# yolov5 학습 모델을 불러줍니다
model, names, device = v5_load('/Users/kangwook/Desktop/Captone/댕댕/exp3/weights/best.pt', mode='cpu')

# colors = setcolor(mode='random')
colors = [(148, 115, 190),(204, 178, 143),(44, 97, 85),(82, 50, 36),(1, 1, 122),(0, 255, 0)]

# color= [(0, 0, 255)]

# d_img와 detection 파일을 저장할 경로를 정해줍니다
savepath2 = '/Users/kangwook/Desktop/Captone/댕댕/result/det/'
savepath1 = '/Users/kangwook/Desktop/Captone/static/'


val2 = ['구진_플라크','비듬_각질_상피성잔고리','태선화_과다색소침착','농포_여드름','미란_궤양','결절_종괴']
val = ['Papule_Plaque', 'Dandruff_Corneous', 'Lichenification', 'Pustule', 'Ulcer', 'Lump']
# imgpath = imgpaths[0]
for imgpath in imgpaths:
    img = imgload(imgpath)
    
    _img = preprocessing(img, model, device)
    txt = v5_det(_img, img, model)
    # 탐지가 잘 됐는지 확인하기 위해 bounding box 그려줍니다
    d_img = yolodraw(img, txt, colors, (0, 1, 2, 3, 4, 5), mode='det')
    
    if len(txt)==0:
        txt = []
    if len(txt)>0:
        # 정해준 저장경로에 d_img파일 저장
        
        imgname = savepath1 + imgpath.split('/')[7]
        load_img = cv2.imread(imgname)
        txt = yolo2voc(img, txt, mode='det')
        
        for j in range(len(txt)):
            class_num = int(txt[j][0])
            x = int(txt[j][2])
            y = int(txt[j][3] - 30)
            color = colors[(int(txt[j][0]))]
            d_text = cv2.putText(d_img, val[class_num], (x,y), cv2.FONT_HERSHEY_SIMPLEX, 1, colors[class_num], 5)

        cv2.imwrite(imgname, d_img)
        
        # # txt
        txtname1 = savepath2 + imgpath.split('/')[7]
        txtname = txtname1.replace('jpg', 'txt')
        txtname = txtname.replace('d_img', 'det')
        
        
        df_txt = pd.DataFrame(txt)
        for k in range(len(df_txt)):
            class_num = int(txt[k][0])
            df_txt[0] = val2[class_num]

        df_txt = df_txt.astype({2: int, 3: int, 4: int, 5: int})
        df_txt.to_csv(txtname, sep = ' ', index = False, header = False)
    
    
    
    