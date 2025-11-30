import cv2
import mediapipe as mp
import numpy as np

# Mediapipe 設定
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# ---------- 面積計算函式 (新增) ----------
def calculate_quadrilateral_area(p15, p17, p21, p19):
    """
    使用鞋帶公式計算四個點 (P15, P17, P19, P21) 形成的四邊形面積。
    點必須是像素坐標 (x, y)。
    
    :param p15: P15 (左手腕) 坐標，(x15, y15)
    :param p17: P17 (左小指) 坐標
    :param p19: P19 (左拇指) 坐標
    :param p21: P21 (左食指) 坐標
    :return: 像素面積
    """
    # 確保點的順序是 P15 -> P17 -> P19 -> P21 (順序可能影響準確性，但通常用於計算手掌面積)
    points = np.array([p15, p17, p19, p21])
    
    x = points[:, 0]
    y = points[:, 1]
    
    # 鞋帶公式: 0.5 * |(x1*y2 + x2*y3 + ... + xn*y1) - (y1*x2 + y2*x3 + ... + yn*x1)|
    sum1 = np.sum(x * np.roll(y, -1)) # x1*y2 + x2*y3 + x3*y4 + x4*y1
    sum2 = np.sum(y * np.roll(x, -1)) # y1*x2 + y2*x3 + y3*x4 + y4*x1
    
    area = 0.5 * np.abs(sum1 - sum2)
    
    return area  
# ------------------------------------------
    
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

        # 將圖像 BGR 轉成 RGB
        img2 = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # 處理圖像
        results = pose.process(img2) 

        try:
            left_hand_area = 0.0 # 初始化面積變數
            
            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark

                # 抓取四個關鍵點像素座標
                p15 = np.array([landmarks[15].x * w, landmarks[15].y * h])
                p17 = np.array([landmarks[17].x * w, landmarks[17].y * h])
                p19 = np.array([landmarks[19].x * w, landmarks[19].y * h])
                p21 = np.array([landmarks[21].x * w, landmarks[21].y * h])

                # ---------- 計算左手部面積 15-17-21-19 (注意順序) ----------
                left_hand_area = calculate_quadrilateral_area(p15, p17, p19, p21)

                cv2.putText(img, f"P15: {p15.astype(int)}, P17: {p17.astype(int)}, P19: {p19.astype(int)}, P21: {p21.astype(int)}",
                            (600, 100), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 255, 0), 2
                )
                
                # 在影像上顯示左手部面積
                cv2.putText(
                    img, f"Left Hand Area: {int(left_hand_area)} px^2",
                    (600, 200), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (255, 0, 0), 2  # 藍色文字
                )

                # 注意：fillPoly 和 polylines 都需要一個頂點陣列的列表
                points_int = np.array([p15, p17, p19, p21], dtype=np.int32).reshape((-1, 1, 2))
                fill_color = (255, 200, 0) # 例子：淡藍色
                cv2.fillPoly(img, [points_int], fill_color)

                left_hand_points = np.array([p15, p17, p19, p21])
                left_hand_center = np.mean(left_hand_points, axis=0) # 計算四點的平均作為中心點
                cv2.putText(img, f"Left Hand Center: ({int(left_hand_center[0])}, {int(left_hand_center[1])})",
                            (600, 250), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 255, 255), 2
                )
                cv2.circle(img, (int(left_hand_center[0]), int(left_hand_center[1])), 8, (0, 0, 255), -1)
                

# 抓取四個關鍵點像素座標
                p16 = np.array([landmarks[16].x * w, landmarks[16].y * h])
                p18 = np.array([landmarks[18].x * w, landmarks[18].y * h])
                p20 = np.array([landmarks[20].x * w, landmarks[20].y * h])
                p22 = np.array([landmarks[22].x * w, landmarks[22].y * h])

                # ---------- 計算左手部面積 15-17-21-19 (注意順序) ----------
                right_hand_area = calculate_quadrilateral_area(p16, p18, p20, p22)

                cv2.putText(img, f"P16: {p16.astype(int)}, P18: {p18.astype(int)}, P20: {p20.astype(int)}, P22: {p22.astype(int)}",
                            (50, 50), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 255, 0), 2
                )
                
                # 在影像上顯示右手部面積
                cv2.putText(
                    img, f"Right Hand Area: {int(right_hand_area)} px^2",
                    (50, 150), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (255, 0, 0), 2  # 藍色文字
                )

                # 注意：fillPoly 和 polylines 都需要一個頂點陣列的列表
                points_int = np.array([p16, p18, p20, p22], dtype=np.int32).reshape((-1, 1, 2))
                fill_color = (255, 200, 0) # 例子：淡藍色
                cv2.fillPoly(img, [points_int], fill_color)
                
                right_hand_points = np.array([p16, p18, p20, p22])
                right_hand_center = np.mean(right_hand_points, axis=0) # 計算四點的平均作為中心點
                cv2.putText(img, f"Right Hand Center: ({int(right_hand_center[0])}, {int(right_hand_center[1])})",
                            (50, 200), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 255, 255), 2
                )
                cv2.circle(img, (int(right_hand_center[0]), int(right_hand_center[1])), 8, (0, 0, 255), -1)

                # """計算手部到嘴角兩點的距離"""
                # 此處可以加入距離計算

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
            # print("Error:", e) # 由於攝影機剛啟動時可能沒有偵測到點，會導致錯誤，先註釋掉
            pass # 這裡使用 pass 來忽略沒有偵測到 landmarks 時的錯誤

cap.release()
cv2.destroyAllWindows()