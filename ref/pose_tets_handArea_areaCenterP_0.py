import cv2
import mediapipe as mp
import numpy as np

# Mediapipe 設定
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# ---------- 面積和重心計算函式 (更新) ----------

def calculate_signed_area_and_centroid(p15, p17, p21, p19):
    """
    使用鞋帶公式計算四個點 (P15, P17, P19, P21) 形成的四邊形的有向面積和重心。
    點必須是像素坐標 (x, y)。
    
    :param p15: P15 (左手腕) 坐標，(x15, y15)
    :param p17: P17 (左小指) 坐標
    :param p21: P21 (左拇指) 坐標
    :param p19: P19 (左食指) 坐標
    :return: (area, centroid_x, centroid_y)
    """
    # 確保點的順序是 P15 -> P17 -> P19 -> P21 (這個順序是您在主程式中使用的)
    # 注意：在主程式中您傳入的順序是 (p15, p17, p19, p21)，但在 calculate_quadrilateral_area 函式的註釋中寫的是 (p15, p17, p21, p19)，
    # 這裡我以主程式實際調用的順序 (p15, p17, p19, p21) 為準。
    points = np.array([p15, p17, p19, p21]) 
    
    x = points[:, 0]
    y = points[:, 1]
    
    # 為了套用多邊形重心公式，我們需要將點循環：(x5, y5) = (x1, y1)
    x_next = np.roll(x, -1) # x2, x3, x4, x1
    y_next = np.roll(y, -1) # y2, y3, y4, y1
    
    # 計算有向面積 (Signed Area)
    # A = 0.5 * sum(xi*y_i+1 - x_i+1*yi)
    signed_area_term = (x * y_next) - (x_next * y)
    signed_area = 0.5 * np.sum(signed_area_term)
    
    area = np.abs(signed_area) # 實際像素面積
    
    # 計算重心 (Centroid)
    if area < 1.0: # 如果面積太小，無法計算重心，直接返回中心點作為近似
        centroid_x = np.mean(x)
        centroid_y = np.mean(y)
    else:
        # Cx = (1 / 6A) * sum((xi + x_i+1) * (xi*y_i+1 - x_i+1*yi))
        centroid_x = (1.0 / (6.0 * signed_area)) * np.sum((x + x_next) * signed_area_term)
        
        # Cy = (1 / 6A) * sum((yi + y_i+1) * (xi*y_i+1 - x_i+1*yi))
        centroid_y = (1.0 / (6.0 * signed_area)) * np.sum((y + y_next) * signed_area_term)

    return area, centroid_x, centroid_y

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
                p15 = np.array([landmarks[15].x * w, landmarks[15].y * h]) # 左手腕
                p17 = np.array([landmarks[17].x * w, landmarks[17].y * h]) # 左小指
                p19 = np.array([landmarks[19].x * w, landmarks[19].y * h]) # 左食指
                p21 = np.array([landmarks[21].x * w, landmarks[21].y * h]) # 左拇指

                # ---------- 計算左手部面積和重心 (使用新的函式) ----------
                left_hand_area, centroid_x, centroid_y = calculate_signed_area_and_centroid(p15, p17, p19, p21)
                
                # 在影像上顯示左手部面積
                cv2.putText(
                    img, f"Left Hand Area: {int(left_hand_area)} px^2",
                    (50, 50), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (255, 0, 0), 2  # 藍色文字
                )
                
                # 顯示重心座標 (可選)
                cv2.putText(
                    img, f"Centroid: ({int(centroid_x)}, {int(centroid_y)})",
                    (50, 150), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0, 255, 0), 2  # 綠色文字
                )
                
                # 在重心位置畫一個點
                cv2.circle(
                    img, 
                    (int(centroid_x), int(centroid_y)), 
                    radius=5, 
                    color=(0, 0, 255), # 紅色
                    thickness=-1 # 實心
                )
                
                '''
                # 注意：fillPoly 和 polylines 都需要一個頂點陣列的列表
                points_int = np.array([p15, p17, p19, p21], dtype=np.int32).reshape((-1, 1, 2))
                fill_color = (255, 200, 0) # 例子：淡藍色
                cv2.fillPoly(img, [points_int], fill_color)
                '''
                
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
            # print("Error:", e) 
            pass # 這裡使用 pass 來忽略沒有偵測到 landmarks 時的錯誤

cap.release()
cv2.destroyAllWindows()