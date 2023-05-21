
from flask import Flask ,request,render_template,redirect,url_for
from werkzeug.utils import secure_filename
import glob,os,json,os.path
import cv2
import numpy as np
import pandas as pd
from info import *;

app = Flask(__name__)
app.config['IMG_FOLDER'] = os.path.join('static', 'images') #이미지 파일의 경로 지정

imgpaths = glob.glob('C:/Users/kangwook/Desktop/Capstone/dang/save/*.jpg')
model, names, device = v5_load('C:/Users/kangwook/Desktop/Capstone/dang/exp3/weights/best.pt', mode='cpu')
colors = setcolor(mode='random')
savepath2 = 'C:/Users/kangwook/Desktop/Capstone/dang/result/det/'
savepath1 = 'C:/Users/kangwook/Desktop/Capstone/yolov5/static/'
val2 = ['구진_플라크','비듬_각질_상피성잔고리','태선화_과다색소침착','농포_여드름','미란_궤양','결절_종괴']
val = ['Papule_Plaque', 'Dandruff_Corneous', 'Lichenification', 'Pustule', 'Ulcer', 'Lump']

'''
def result():
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
            
            imgname = savepath1 + imgpath.split('\\')[1]
            txt = yolo2voc(img, txt, mode='det')
            
            for j in range(len(txt)):
                class_num = int(txt[j][0])
                x = int(txt[j][2])
                y = int(txt[j][3] - 30)
                color = colors[(int(txt[j][0]))]
                d_text = cv2.putText(d_img, val[class_num], (x,y), cv2.FONT_HERSHEY_SIMPLEX, 1, colors[class_num], 5)

            cv2.imwrite(imgname, d_img)
            
            # # txt
            txtname1 = savepath2 + imgpath.split('\\')[1]
            txtname = txtname1.replace('jpg', 'txt')
            txtname = txtname.replace('d_img', 'det')
            
            df_txt = pd.DataFrame(txt)
            for k in range(len(df_txt)):
                class_num = int(txt[k][0])
                df_txt[0] = val2[class_num]

            df_txt = df_txt.astype({2: int, 3: int, 4: int, 5: int})
            df_txt.to_csv(txtname, sep = ' ', index = False, header = False)
            
  '''          
@app.route('/fileupload', methods=['GET','POST'])
def fileupload():
    if request.method == 'POST':
        f = request.files['file']
        f.save('C:/Users/kangwook/Desktop/Capstone/dang/save/' + secure_filename(f.filename))
        return redirect('/sender')
    else:
        return render_template('fileupload.html')
        
       
    
@app.route('/sender',methods=['GET'])
def sender():
    imgpaths = glob.glob('C:/Users/kangwook/Desktop/Capstone/dang/save/*.jpg')
    model, names, device = v5_load('C:/Users/kangwook/Desktop/Capstone/dang/exp3/weights/best.pt', mode='cpu')
    colors = setcolor(mode='random')
    savepath2 = 'C:/Users/kangwook/Desktop/Capstone/dang/result/det/'
    savepath1 = 'C:/Users/kangwook/Desktop/Capstone/yolov5/static/'
    val2 = ['구진_플라크','비듬_각질_상피성잔고리','태선화_과다색소침착','농포_여드름','미란_궤양','결절_종괴']
    val = ['Papule_Plaque', 'Dandruff_Corneous', 'Lichenification', 'Pustule', 'Ulcer', 'Lump']
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
            imgname = savepath1 + imgpath.split('\\')[1]
            txt = yolo2voc(img, txt, mode='det')
            
            for j in range(len(txt)):
                class_num = int(txt[j][0])
                x = int(txt[j][2])
                y = int(txt[j][3] - 30)
                color = colors[(int(txt[j][0]))]
                d_text = cv2.putText(d_img, val[class_num], (x,y), cv2.FONT_HERSHEY_SIMPLEX, 1, colors[class_num], 5)
            cv2.imwrite(imgname, d_img)
            # # txt
            txtname1 = savepath2 + imgpath.split('\\')[1]
            txtname = txtname1.replace('jpg', 'txt')
            txtname = txtname.replace('d_img', 'det')
            df_txt = pd.DataFrame(txt)
            for k in range(len(df_txt)):
                class_num = int(txt[k][0])
                df_txt[0] = val2[class_num]
            df_txt = df_txt.astype({2: int, 3: int, 4: int, 5: int})
            df_txt.to_csv(txtname, sep = ' ', index = False, header = False)
    return redirect('/showimg')
    
    
    
    
    

@app.route('/showimg',methods=['GET'])
def showimg():
    list_of_files = glob.glob('C:/Users/kangwook/Desktop/Capstone/dang/result/det/*')
    latest_file = max(list_of_files, key=os.path.getctime)
    with open(latest_file, encoding='UTF-8') as f:
        lines = f.readline()
        result = lines.split(' ')
        col = ['병명','스코어','x좌표', 'y좌표', 'w좌표', 'h좌표']
        result = dict(zip(col,result))
        info = json.dumps(result,ensure_ascii=False)
    # list_of_files2 = glob.glob('/Users/kangwook/Desktop/Captone/static/*.jpg')
    list_of_files2 = glob.glob('C:/Users/kangwook/Desktop/Capstone/yolov5/static/*.jpg')
    latest_file2 = max(list_of_files2, key=os.path.getctime)
    result = latest_file2.split('\\')[1]
    return render_template("index.html",img_file=result,info=info)



    






    
        
            
if __name__ == "__main__": 
     app.run(host='localhost', port='5001', debug=False)
