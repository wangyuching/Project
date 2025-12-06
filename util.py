import cv2
import mediapipe as mp
import numpy as np

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

''' ------ 座標轉換 (將正規化座標 [0, 1] 轉換為實際像素座標 [0*w, 1*h]) ------ '''
def get_coords(landmark, w, h):
    return np.array([landmark.x * w, landmark.y * h])

def calculate_quadrilateral_area(p15, p17, p21, p19):
    """
    鞋帶公式順時針計算四個點 (P15, P17, P19, P21) 形成的四邊形面積。
    """
    points = np.array([p15, p17, p19, p21])
    
    x = points[:, 0]
    y = points[:, 1]
    
    sum1 = np.sum(x * np.roll(y, -1)) 
    sum2 = np.sum(y * np.roll(x, -1)) 
    
    area = 0.5 * np.abs(sum1 - sum2)
    
    return area 

def calculate_angle(a, b, c):
    """
    計算三點 a-b-c 的夾角（以 b 為頂點）
    a, b, c: numpy array 座標
    回傳角度 (degree)
    """
    ba = a - b
    bc = c - b

    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    cosine_angle = np.clip(cosine_angle, -1.0, 1.0)

    angle = np.degrees(np.arccos(cosine_angle))
    return angle

def frame(img, results, w, h):

    left_elbow_angle = 0
    right_elbow_angle = 0
    left_distance = 0
    right_distance = 0

    try:        
        if not results.pose_landmarks:
            return img, left_elbow_angle, right_elbow_angle, left_distance, right_distance
            
        landmarks = results.pose_landmarks.landmark

        ''' ---------- 左手部landmark 左肩[11]-左肘[13]-左腕[15], 左腕[15]-左小指[17]-左食指[19]-左姆指[21] ---------- '''
        p11 = get_coords(landmarks[11], w, h) #左肩
        p13 = get_coords(landmarks[13], w, h) #左肘
        p15 = get_coords(landmarks[15], w, h) #左腕
        p17 = get_coords(landmarks[17], w, h) #左小指
        p19 = get_coords(landmarks[19], w, h) #左食指
        p21 = get_coords(landmarks[21], w, h) #左姆指

        ''' ---------- 右手部landmark 右肩[12]-右肘[14]-右腕[16], 右腕[16]-右小指[18]-右食指[20]-右姆指[22] ---------- '''
        p12 = get_coords(landmarks[12], w, h) #右肩
        p14 = get_coords(landmarks[14], w, h) #右肘
        p16 = get_coords(landmarks[16], w, h) #右腕
        p18 = get_coords(landmarks[18], w, h) #右小指
        p20 = get_coords(landmarks[20], w, h) #右食指
        p22 = get_coords(landmarks[22], w, h) #右姆指
        
        ''' ---------- 計算左手部四點的平均作為中心點 ---------- '''
        left_hand_points = np.array([p15, p17, p19, p21])
        left_hand_center = np.mean(left_hand_points, axis=0)

        # ''' ---------- 計算左手部面積 15-17-21-19 (注意順序) ---------- '''
        # left_hand_area = calculate_quadrilateral_area(p15, p17, p19, p21)
        ''' ---------- 顯示左手部面積 BGR ---------- '''
        points_int = np.array([p15, p17, p19, p21], dtype=np.int32).reshape((-1, 1, 2))
        fill_color = (255, 200, 0)
        cv2.fillPoly(img, [points_int], fill_color)

        
        # ''' ---------- 顯示左手部中心點座標 ---------- '''
        # cv2.putText(img, f"Left Hand Center: ({int(left_hand_center[0])}, {int(left_hand_center[1])})", 
        #             (670, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        ''' ---------- 顯示左手部中心點 ---------- '''
        cv2.circle(img, (int(left_hand_center[0]), int(left_hand_center[1])), 8, (0, 255, 255), -1)


        ''' ---------- 右手部四點 16-18-20-22 ---------- '''
        p16 = np.array([landmarks[16].x * w, landmarks[16].y * h])
        p18 = np.array([landmarks[18].x * w, landmarks[18].y * h])
        p20 = np.array([landmarks[20].x * w, landmarks[20].y * h])
        p22 = np.array([landmarks[22].x * w, landmarks[22].y * h])
        # ''' ---------- 計算右手部面積 16-18-20-22 (注意順序) ---------- '''
        # right_hand_area = calculate_quadrilateral_area(p16, p18, p20, p22)
        ''' ---------- 顯示右手部面積 BGR ---------- '''
        points_int = np.array([p16, p18, p20, p22], dtype=np.int32).reshape((-1, 1, 2))
        fill_color = (255, 0, 0)
        cv2.fillPoly(img, [points_int], fill_color)

        ''' ---------- 計算右手部四點的平均作為中心點 ---------- '''
        right_hand_points = np.array([p16, p18, p20, p22])
        right_hand_center = np.mean(right_hand_points, axis=0)
        # ''' ---------- 顯示右手部中心點座標 ---------- '''
        # cv2.putText(img, f"Right Hand Center: ({int(right_hand_center[0])}, {int(right_hand_center[1])})",
        #             (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (125, 0, 255), 2)
        ''' ---------- 顯示右手部中心點 ---------- '''
        cv2.circle(img, (int(right_hand_center[0]), int(right_hand_center[1])), 8, (125, 0, 255), -1)


        ''' ---------- 嘴角二點 9-10 ---------- '''
        p9 = np.array([landmarks[9].x * w, landmarks[9].y * h])
        p10 = np.array([landmarks[10].x * w, landmarks[10].y * h])

        ''' ---------- 計算嘴角二點中心點 ---------- '''
        mouth_center_point = (p9 + p10) / 2
        # ''' ---------- 顯示嘴角二點中心點座標 ---------- '''
        # cv2.putText(img, f"Center Mouth Point: ({int(mouth_center_point[0])}, {int(mouth_center_point[1])})",
        #             (450, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        ''' ---------- 顯示嘴角二點中心點 ---------- '''
        cv2.circle(img, (int(mouth_center_point[0]), int(mouth_center_point[1])), 8, (0, 255, 0), -1)


        """ ------ 左手部中心點到嘴角二點中心點的距離 ------ """
        left_distence = np.linalg.norm(left_hand_center - mouth_center_point)
        cv2.putText(img, f"Left Hand to Mouth Distance: {int(left_distence)} px",
                    (670, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        # cv2.line(img, (int(left_hand_center[0]), int(left_hand_center[1])), 
        #             (int(mouth_center_point[0]), int(mouth_center_point[1])),
        #             (0, 255, 255), 2)

        """ ------ 右手部中心點到嘴角二點中心點的距離 ------ """
        right_distence = np.linalg.norm(right_hand_center - mouth_center_point)
        cv2.putText(img, f"Right Hand to Mouth Distance: {int(right_distence)} px",
                    (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (125, 0, 255), 2)
        # cv2.line(img, (int(right_hand_center[0]), int(right_hand_center[1])), 
        #             (int(mouth_center_point[0]), int(mouth_center_point[1])),
        #             (125, 0, 255), 2)


        ''' ------ 左手肘角度 11-13-15 ------ '''
        p11 = np.array([landmarks[11].x * w, landmarks[11].y * h])
        p13 = np.array([landmarks[13].x * w, landmarks[13].y * h])
        p15 = np.array([landmarks[15].x * w, landmarks[15].y * h])

        ''' ---------- 計算左手肘角度 ---------- '''
        left_elbow_angle = calculate_angle(p11, p13, p15)

        ''' ---------- 顯示左手肘角度 ---------- '''
        cv2.putText(
            img, f"Left Elbow: {int(left_elbow_angle)} deg",
            (670, 50), cv2.FONT_HERSHEY_SIMPLEX,
            1, (0, 255, 255), 2
        )

        ''' ---------- 計算右手肘角度 12-14-16 ---------- '''
        p12 = np.array([landmarks[12].x * w, landmarks[12].y * h])
        p14 = np.array([landmarks[14].x * w, landmarks[14].y * h])
        p16 = np.array([landmarks[16].x * w, landmarks[16].y * h])

        ''' ---------- 計算右手肘角度 ---------- '''
        right_elbow_angle = calculate_angle(p12, p14, p16)

        ''' ---------- 顯示右手肘角度 ---------- '''
        cv2.putText(
            img, f"Right Elbow: {int(right_elbow_angle)} deg",
            (50, 50), cv2.FONT_HERSHEY_SIMPLEX,
            1, (0, 0, 255), 2
        )

    except Exception as e:
        print("Error:", e)


