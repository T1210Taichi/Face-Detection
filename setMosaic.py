#顔を検出し、顔の位置に画像を張り付ける

import cv2
import numpy

def mosaic(img, scale=0.1):
    h, w = img.shape[:2]  # 画像の大きさ

    # 画像を scale (0 < scale <= 1) 倍に縮小する。
    dst = cv2.resize(
        img, dsize=None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST
    )

    # 元の大きさに拡大する。
    dst = cv2.resize(dst, dsize=(w, h), interpolation=cv2.INTER_NEAREST)

    return dst

#カスケード
cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

# VideoCapture オブジェクトを取得します
capture = cv2.VideoCapture(0)

while(True):
    ret, frame = capture.read()
    frame_height,frame_width = frame.shape[:2]

    #顔の学習データ精査
    front_face_list=cascade.detectMultiScale(frame,scaleFactor=1.1,minNeighbors=3,minSize=(90,90))
    #print(front_face_list)

    #画像を当てはめる
    if len(front_face_list)>0:
        for (x,y,w,h) in front_face_list:
            #顔のモザイク
            roi = frame[y:y+h,x:x+w]
            roi[:] = mosaic(roi)
            #当てはめる
            frame[y:y+h,x:x+w] = roi
            cv2.imshow('frame',frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

capture.release()
cv2.destroyAllWindows()
