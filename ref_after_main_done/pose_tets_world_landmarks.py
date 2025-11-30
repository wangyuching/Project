import cv2
import mediapipe as mp
import numpy as np

# Mediapipe 設定
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# 夾角計算函式 (改為接受 3D 座標)
def calculate_angle(a, b, c):
    """
    計算三點 a-b-c 的夾角（以 b 為頂點）
    a, b, c: numpy array 座標 (x, y, z)
    回傳角度（degree）
    """
    ba = a - b
    bc = c - b

    # 使用點積和向量長度計算餘弦值
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    # 限制餘弦值在 [-1.0, 1.0] 之間，避免浮點數誤差導致的 arccos 錯誤
    cosine_angle = np.clip(cosine_angle, -1.0, 1.0)

    angle = np.degrees(np.arccos(cosine_angle))
    return angle


# 開啟攝影機
cap = cv2.VideoCapture(0)

# 使用 Pose 模組，並開啟 World Landmarks 輸出
with mp_pose.Pose(min_detection_confidence=0.5,min_tracking_confidence=0.5) as pose:

    # 保持畫面對比度與清晰度
    w, h = 1200, 800 

    while cap.isOpened():
        ret, frame = cap.read()
        img = cv2.resize(frame, (w, h))
        if not ret:
            print("Ignoring empty camera frame.")
            break

        try:
            img2 = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = pose.process(img2)

            # 確保偵測到身體地標
            if results.pose_world_landmarks:
                # *** 關鍵變動：使用世界地標 (World Landmarks) ***
                world_landmarks = results.pose_world_landmarks.landmark

                # ---------- 計算左手肘角度 (11-13-15) - 使用 3D 座標 ----------
                
                # 抓取三個關鍵點的 3D 世界座標 (x, y, z)，單位：公尺
                # 11: 左肩 (Left Shoulder)
                # 13: 左手肘 (Left Elbow)
                # 15: 左手腕 (Left Wrist)
                
                wl_11 = np.array([world_landmarks[11].x, world_landmarks[11].y, world_landmarks[11].z])
                wl_13 = np.array([world_landmarks[13].x, world_landmarks[13].y, world_landmarks[13].z])
                wl_15 = np.array([world_landmarks[15].x, world_landmarks[15].y, world_landmarks[15].z])

                # 使用自訂函式計算 3D 角度
                angle_left_elbow_3d = calculate_angle(wl_11, wl_13, wl_15)

                # 畫角度
                cv2.putText(
                    img, f"3D Left Elbow: {int(angle_left_elbow_3d)} deg",
                    (450, 50), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0, 255, 0), 2
                )
                
                # ---------- 計算右手肘角度 (12-14-16) - 使用 3D 座標 ----------
                
                # 12: 右肩 (Right Shoulder)
                # 14: 右手肘 (Right Elbow)
                # 16: 右手腕 (Right Wrist)
                wl_12 = np.array([world_landmarks[12].x, world_landmarks[12].y, world_landmarks[12].z])
                wl_14 = np.array([world_landmarks[14].x, world_landmarks[14].y, world_landmarks[14].z])
                wl_16 = np.array([world_landmarks[16].x, world_landmarks[16].y, world_landmarks[16].z])

                # 使用自訂函式計算 3D 角度
                angle_right_elbow_3d = calculate_angle(wl_12, wl_14, wl_16)

                # 畫角度
                cv2.putText(
                    img, f"3D Right Elbow: {int(angle_right_elbow_3d)} deg",
                    (30, 50), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0, 0, 255), 2
                )

                # -------------------------------------------------------------
                # 為了維持視覺化，我們還是使用正規化地標來繪製骨架
                landmarks = results.pose_landmarks.landmark
                mp_drawing.draw_landmarks(
                    img,
                    results.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=4),
                    mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
                )

            # 顯示畫面
            cv2.imshow("Webcam Feed (3D Angle Analysis)", img)

            # 按 q 離開
            if cv2.waitKey(10) == ord('q'):
                break

        except Exception as e:
            print("Error:", e)
            break

cap.release()
cv2.destroyAllWindows()