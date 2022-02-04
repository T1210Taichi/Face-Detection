import cv2
import math
import numpy as np
import mediapipe as mp

# VideoCapture オブジェクトを取得します
capture = cv2.VideoCapture(0)

#hand
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

while(True):
    ret, frame = capture.read()
    #あらかじめ左右反転
    annotated_image = np.fliplr(frame)

    images = {'face':frame}

    # Run MediaPipe Hands.
    with mp_hands.Hands(
        static_image_mode=True,
        max_num_hands=2,
        min_detection_confidence=0.7) as hands:
        for name, image in images.items():
            # Convert the BGR image to RGB, flip the image around y-axis for correct 
            # handedness output and process it with MediaPipe Hands.
            #results = hands.process(cv2.flip(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), 1))
            results = hands.process(cv2.flip(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), 1))

            # Print handedness (left v.s. right hand).
            print(f'Handedness of {name}:')
            print(results.multi_handedness)

            if not results.multi_hand_landmarks:
                continue
            # Draw hand landmarks of each hand.
            print(f'Hand landmarks of {name}:')
            image_hight, image_width, _ = image.shape
            annotated_image = cv2.flip(image.copy(), 1)
            for hand_landmarks in results.multi_hand_landmarks:
                # Print index finger tip coordinates.
                print(
                    f'Index finger tip coordinate: (',
                    f'{hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x * image_width}, '
                    f'{hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y * image_hight})'
                )
                mp_drawing.draw_landmarks(
                    annotated_image,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style())


    cv2.imshow('frame',annotated_image)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

capture.release()
cv2.destroyAllWindows()