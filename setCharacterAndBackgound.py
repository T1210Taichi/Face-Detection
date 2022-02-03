#顔を検出し、顔の位置に画像を張り付ける

import cv2
from numpy import character

#カスケード
cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

#キャラクター
character = cv2.imread("character.png")
character = cv2.resize(character,dsize=(250,250))
chr_height, chr_width = character.shape[:2]

#背景
background_def = cv2.imread("background.png")
back_height, background_width = background_def.shape[:2]

# VideoCapture オブジェクトを取得します
capture = cv2.VideoCapture(0)

while(True):
    ret, frame = capture.read()
    frame_height,frame_width = frame.shape[:2]

    #背景をリサイズ
    background = cv2.resize(background_def,dsize=(frame_width,frame_height))

    #顔の学習データ精査
    front_face_list=cascade.detectMultiScale(frame,scaleFactor=1.1,minNeighbors=3,minSize=(100,100))
    #print(front_face_list)

    #画像を当てはめる
    for (x,y,w,h) in front_face_list:
        if x + chr_width < frame_width and y + chr_height < frame_height:
            background[y:y+chr_height,x:x+chr_width] = character
            cv2.imshow('frame',background)  

    background = background_def

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

capture.release()
cv2.destroyAllWindows()