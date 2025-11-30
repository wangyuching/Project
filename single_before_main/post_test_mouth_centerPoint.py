import cv2
import mediapipe as mp
import numpy as np

# Mediapipe 設定
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# 開啟攝影機
cap = cv2.VideoCapture(0)

with mp_pose.Pose(min_detection_confidence=0.5,min_tracking_confidence=0.5) as pose:

    w, h = 1200, 800

    while cap.isOpened():
        ret, frame = cap.read()
        img = cv2.resize(frame, (w, h))
        if not ret:
            print("Ignoring empty camera frame.")
            break

        try:
            img2 = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) #將圖像(攝影機格式) BGR 轉成(2=to) RGB(mediapipe格式) 後存在 img2。
            results = pose.process(img2) # 啟動Mediapipe身體偵測模組 > 處理 img2 (分析身體) > 偵測 21 個關鍵點 > 回傳結果

            # pose.landmarks: https://chuoling.github.io/mediapipe/images/mobile/pose_tracking_full_body_landmarks.png
            # ---------- 計算嘴角中心點 9-10----------
            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark

                # 抓取三個關鍵點像素座標
                p9 = np.array([landmarks[9].x * w, landmarks[9].y * h])
                p10 = np.array([landmarks[10].x * w, landmarks[10].y * h])

                center_point = (p9 + p10) / 2

                cv2.putText(img, f"Center Mouth Point: ({int(center_point[0])}, {int(center_point[1])})",
                            (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                # 畫出嘴中心點
                cv2.circle(img, (int(center_point[0]), int(center_point[1])), 8, (0, 255, 0), -1)

                """計算手部到嘴角兩點的距離"""

            # ---------- 畫 Pose.landmarks ----------
            mp_drawing.draw_landmarks(
                img,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=4),
                mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
            )

            # 顯示畫面
            cv2.imshow("Webcam Feed", img)

            # 按 q 離開
            if cv2.waitKey(10) == ord('q'):
                break

        except Exception as e:
            print("Error:", e)
            break

cap.release()
cv2.destroyAllWindows()