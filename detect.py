import cv2
from numpy import character

#カスケード
cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

# VideoCapture オブジェクトを取得します
capture = cv2.VideoCapture(0)

while(True):
    ret, frame = capture.read()

    #グレースケール変換
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

    #顔の学習データ精査
    front_face_list=cascade.detectMultiScale(gray,scaleFactor=1.1,minNeighbors=3,minSize=(90,90))
    print(front_face_list)

    #顔を囲む
    for (x,y,w,h) in front_face_list:
        cv2.rectangle(gray,(x,y),(x+w,y+h),(255,0,255),thickness=5)
        cv2.imshow('frame',gray)  

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

capture.release()
cv2.destroyAllWindows()