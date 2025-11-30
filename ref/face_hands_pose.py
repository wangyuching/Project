import mediapipe as mp
import cv2

mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic
#holistic: 同時偵測並連接「臉部（face mesh）」「身體姿態（pose）」和「雙手（left/right hand）」的關鍵點（landmarks）
mp_face_mesh = mp.solutions.face_mesh
mp_hands = mp.solutions.hands
mp_pose = mp.solutions.pose

cap = cv2.VideoCapture(0)
# 啟動並設定 Holistic 模型. detection:偵測, tracking:追蹤
with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    w, h = 800, 600
    while cap.isOpened():
        ret, frame = cap.read()

        img = cv2.resize(frame, (w, h))
        if not ret:
            print("Ignoring empty camera frame.")
            break

        try:
            # Recolor Feed:let holistic model is RGB. OpenCV's BGR train to mediapipe's RGB
            img2 = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # 啟動Mediapipe Holistic偵測模組 > 處理 img2 > 同時偵測並追蹤人體臉部(Face_mesh)、雙手(hands)、身體姿態(Pose) 關鍵點 > 回傳結果
            results = holistic.process(img2)

            # 1. Draw face landmarks
            mp_drawing.draw_landmarks(
                img, #顯示在 img 上,OpenCV格式:BGR
                results.face_landmarks , #標示臉部 468 個關鍵點, holistic model
                mp_face_mesh.FACEMESH_TESSELATION, #臉部關鍵點連線方式『_TESSELATION』, fece mesh model
                mp_drawing.DrawingSpec(color=(80,110,10), thickness=1, circle_radius=1),
                mp_drawing.DrawingSpec(color=(80,256,121), thickness=1, circle_radius=1)
                )
            
            # 2. Right hand
            mp_drawing.draw_landmarks(
                img, 
                results.right_hand_landmarks, #標示右手 21 個關鍵點
                mp_hands.HAND_CONNECTIONS, #雙手關鍵點連線方式『_CONNECTIONS』
                mp_drawing.DrawingSpec(color=(80,22,10), thickness=2, circle_radius=4),
                mp_drawing.DrawingSpec(color=(80,44,121), thickness=2, circle_radius=2)
                )

            # 3. Left Hand
            mp_drawing.draw_landmarks(
                img, 
                results.left_hand_landmarks, #標示左手 21 個關鍵點
                mp_hands.HAND_CONNECTIONS, #雙手關鍵點連線方式『_CONNECTIONS』
                mp_drawing.DrawingSpec(color=(121,22,76), thickness=2, circle_radius=4),
                mp_drawing.DrawingSpec(color=(121,44,250), thickness=2, circle_radius=2)
                                    )

            # 4. Pose Detections
            mp_drawing.draw_landmarks(
                img, 
                results.pose_landmarks, #標示身體姿態 33 個關鍵點
                mp_pose.POSE_CONNECTIONS, #身體姿態關鍵點連線方式『_CONNECTIONS』
                mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=4),
                mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
                )
            
            #即時顯示畫面,OpenCV格式:BGR
            cv2.imshow('Raw Webcam Feed', img)

            #按下q(可以自定)鍵退出while。1:每隔1毫秒檢查一次按鍵是否被按下。
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

        #出錯時印出錯誤訊息並跳出迴圈
        except Exception as e:
            print(e)
            break

cap.release()
cv2.destroyAllWindows()