#顔を検出し、顔の位置に画像を張り付ける

import cv2
from numpy import character

#カスケード
cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

#キャラクター
character = cv2.imread("character.png")
character = cv2.resize(character,dsize=(250,250))
character = cv2.cvtColor(character,cv2.COLOR_BGR2GRAY)
#cv2.imshow('frame',character) 
chr_height, chr_width = character.shape[:2]

# VideoCapture オブジェクトを取得します
capture = cv2.VideoCapture(0)

while(True):
    ret, frame = capture.read()

    #グレースケール変換
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    gray_height,gray_widht = gray.shape[:2]
    #print(gray_height)
    #print(gray_widht)


    #顔の学習データ精査
    front_face_list=cascade.detectMultiScale(gray,scaleFactor=1.1,minNeighbors=3,minSize=(90,90))
    #print(front_face_list)

    #画像を当てはめる
    for (x,y,w,h) in front_face_list:
        if x + chr_width < gray_widht and y + chr_height < gray_height:
            gray[y:y+chr_height,x:x+chr_width] = character
            cv2.imshow('frame',gray)  

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

capture.release()
cv2.destroyAllWindows()