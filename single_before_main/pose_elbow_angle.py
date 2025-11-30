import cv2
import mediapipe as mp
import numpy as np

# Mediapipe 設定
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# 夾角計算函式
def calculate_angle(a, b, c):
    """
    計算三點 a-b-c 的夾角（以 b 為頂點）
    a, b, c: numpy array 座標
    回傳角度（degree）
    """
    ba = a - b
    bc = c - b

    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    cosine_angle = np.clip(cosine_angle, -1.0, 1.0)

    angle = np.degrees(np.arccos(cosine_angle))
    return angle

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
            # ---------- 計算左手肘角度 11-13-15 ----------
            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark

                # 抓取三個關鍵點像素座標
                p11 = np.array([landmarks[11].x * w, landmarks[11].y * h])
                p13 = np.array([landmarks[13].x * w, landmarks[13].y * h])
                p15 = np.array([landmarks[15].x * w, landmarks[15].y * h])

                # 使用自訂函式計算角度
                angle_left_elbow = calculate_angle(p11, p13, p15)

                # 畫角度
                cv2.putText(
                    img, f"Left Elbow: {int(angle_left_elbow)} deg",
                    (450, 50), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0, 255, 0), 2
                )

                # ---------- 計算右手肘角度 12-14-16 ----------
                # 抓取三個關鍵點像素座標
                p12 = np.array([landmarks[12].x * w, landmarks[12].y * h])
                p14 = np.array([landmarks[14].x * w, landmarks[14].y * h])
                p16 = np.array([landmarks[16].x * w, landmarks[16].y * h])

                # 使用自訂函式計算角度
                angle_right_elbow = calculate_angle(p12, p14, p16)

                # 畫角度
                cv2.putText(
                    img, f"Right Elbow: {int(angle_right_elbow)} deg",
                    (30, 50), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0, 0, 255), 2
                )

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