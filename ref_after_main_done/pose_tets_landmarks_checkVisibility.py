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

# --- 🎯 儲存上一次可信賴的座標和角度 ---
# 初始化座標為 NaN，代表還沒有可信賴的值
p11_cache = np.array([np.nan, np.nan])
p13_cache = np.array([np.nan, np.nan])
p15_cache = np.array([np.nan, np.nan])

p12_cache = np.array([np.nan, np.nan])
p14_cache = np.array([np.nan, np.nan])
p16_cache = np.array([np.nan, np.nan])

# --- 🎯 新增: 角度文字緩存 ---
left_elbow_text = "None"
right_elbow_text = "None"


with mp_pose.Pose(min_detection_confidence=0.5,min_tracking_confidence=0.5) as pose:

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

            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark
                
                # 初始化標誌，用於判斷是否成功計算角度
                left_angle_calculated = False
                right_angle_calculated = False

                # ---------- 計算左手肘角度 11-13-15 ----------
                
                # 抓取當前座標
                p11_current = np.array([landmarks[11].x * w, landmarks[11].y * h])
                p13_current = np.array([landmarks[13].x * w, landmarks[13].y * h])
                p15_current = np.array([landmarks[15].x * w, landmarks[15].y * h])
                
                # 檢查三個點的 visibility
                visibility_11 = landmarks[11].visibility
                visibility_13 = landmarks[13].visibility
                visibility_15 = landmarks[15].visibility
                
                # 判斷所有點是否都可信賴 (visibility >= 0.5)
                if visibility_11 >= 0.5 and visibility_13 >= 0.5 and visibility_15 >= 0.5:
                    # 可信賴: 使用當前座標並更新緩存
                    p11, p13, p15 = p11_current, p13_current, p15_current
                    p11_cache, p13_cache, p15_cache = p11_current, p13_current, p15_current
                    
                    angle_left_elbow = calculate_angle(p11, p13, p15)
                    left_elbow_text = f"{int(angle_left_elbow)} deg"
                    left_angle_calculated = True

                else:
                    # 不可信賴: 使用上次緩存的座標 (如果緩存有值)
                    if not np.isnan(p11_cache).any():
                        p11, p13, p15 = p11_cache, p13_cache, p15_cache
                        angle_left_elbow = calculate_angle(p11, p13, p15)
                        left_elbow_text = f"{int(angle_left_elbow)} deg (Cached)" # 顯示為緩存值
                        left_angle_calculated = True
                    else:
                        # 緩存也無值，顯示 None
                        left_elbow_text = "None"


                # ---------- 計算右手肘角度 12-14-16 ----------
                
                # 抓取當前座標
                p12_current = np.array([landmarks[12].x * w, landmarks[12].y * h])
                p14_current = np.array([landmarks[14].x * w, landmarks[14].y * h])
                p16_current = np.array([landmarks[16].x * w, landmarks[16].y * h])

                # 檢查三個點的 visibility
                visibility_12 = landmarks[12].visibility
                visibility_14 = landmarks[14].visibility
                visibility_16 = landmarks[16].visibility

                # 判斷所有點是否都可信賴 (visibility >= 0.5)
                if visibility_12 >= 0.5 and visibility_14 >= 0.5 and visibility_16 >= 0.5:
                    # 可信賴: 使用當前座標並更新緩存
                    p12, p14, p16 = p12_current, p14_current, p16_current
                    p12_cache, p14_cache, p16_cache = p12_current, p14_current, p16_current

                    angle_right_elbow = calculate_angle(p12, p14, p16)
                    right_elbow_text = f"{int(angle_right_elbow)} deg"
                    right_angle_calculated = True
                else:
                    # 不可信賴: 使用上次緩存的座標 (如果緩存有值)
                    if not np.isnan(p12_cache).any():
                        p12, p14, p16 = p12_cache, p14_cache, p16_cache
                        angle_right_elbow = calculate_angle(p12, p14, p16)
                        right_elbow_text = f"{int(angle_right_elbow)} deg (Cached)" # 顯示為緩存值
                        right_angle_calculated = True
                    else:
                        # 緩存也無值，顯示 None
                        right_elbow_text = "None"

                # 繪製地標
                mp_drawing.draw_landmarks(
                    img,
                    results.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=4),
                    mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
                )

            # --- 🎯 新增: 如果沒有偵測到 landmarks，設置為 None ---
            else:
                left_elbow_text = "None"
                right_elbow_text = "None"


            # --- 🎯 統一在外部繪製文字 ---
            # 畫左手肘角度文字
            cv2.putText(
                img, f"Left Elbow: {left_elbow_text}",
                (450, 100), cv2.FONT_HERSHEY_SIMPLEX,
                1, (0, 255, 0), 2
            )
            # 畫右手肘角度文字
            cv2.putText(
                img, f"Right Elbow: {right_elbow_text}",
                (30, 50), cv2.FONT_HERSHEY_SIMPLEX,
                1, (0, 0, 255), 2
            )

            # 顯示畫面
            cv2.imshow("Webcam Feed", img)

            # 按 q 離開
            if cv2.waitKey(10) == ord('q'):
                break

        except Exception as e:
            # 偵測失敗或發生其他錯誤時，顯示 None
            left_elbow_text = "None"
            right_elbow_text = "None"
            # print("Error:", e)
            break

cap.release()
cv2.destroyAllWindows()