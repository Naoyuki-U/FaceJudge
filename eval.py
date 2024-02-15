import cv2
import numpy as np
from keras.models import load_model

heigth=128
width=128

class_names = ['なおゆき', 'しゅり', 'せいや', '粗品']

def evaluation(img_path, model_path):

    answerList = []
    percentList = []
    face_detect_img_pathFull = ""
    target_img_pathList = []
    i = 0
    face_detect_img_path = "./static/images/face_detect/face_detect_"
    target_img_path = "./static/images/cut_dace/cut_dace_"
    ext = ".png"

    model = load_model(model_path) #モデルと重みを復元

    # 学習済みモデルの読み込み
    cascade = cv2.CascadeClassifier("./haarcascade_frontalface_default.xml")

    # 画像データ読み込み
    fname_color = cv2.imread(img_path)
    # 画像データをグレースケール化（白黒）
    fname_gray = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    # 顔を検出する
    lists = cascade.detectMultiScale(fname_gray, minSize=(32, 32))


    # forですべての顔を赤い長方形で囲む
    for (x,y,w,h) in lists:
        cv2.rectangle(fname_color, (x,y), (x+w, y+h), (0, 0, 255), thickness=2)
        face_detect_img_pathFull = f"{face_detect_img_path}{str(i)}{ext}"
        cv2.imwrite(face_detect_img_pathFull, fname_color)
        fname_color_cut = cv2.resize((fname_color[y:y+h, x:x+w]), (heigth, width))
        target_img_pathFull = f"{target_img_path}{str(i)}{ext}"
        cv2.imwrite(target_img_pathFull, fname_color_cut)
        target_img_pathList.append(target_img_pathFull)


        fname_gray_cut = cv2.resize((fname_gray[y:y+h, x:x+w]), (heigth, width))
        data_expanded = np.expand_dims(fname_gray_cut,axis=0)
        cut_images = data_expanded.reshape(data_expanded.shape[0], heigth, width, 1)
        cut_images = cut_images / 255.0

        predictions = model.predict(cut_images)

        answer = class_names[np.argmax(predictions[0])]
        answerList.append(answer)
        percent = 100*np.max(predictions[0])
        percentList.append(percent)
        # print(f"この顔は {answer} です。")
        # print(f"確率: {percent}")
        i += 1

    # 判定結果と加工した画像のpathを返す
    return [answerList, percentList, target_img_pathList], face_detect_img_pathFull, i