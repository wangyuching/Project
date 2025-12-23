import cv2
import mediapipe as mp
import numpy as np

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

def calculate_angle_3d(a, b, c):
    """
    計算肩膀(a) -> 手肘(b) -> 手腕(c) 的連貫向量夾角
    a: 肩膀, b: 手肘, c: 手腕
    夾角公式:
    > 點積(內積) / 範數(向量長度)
    > 反餘弦(acos) 
    > 轉成角度(degress)
    """
    # 向量
    v1 = a - b
    v2 = c - b

    # 點積(內積) / 範數(向量長度)
    denom = np.linalg.norm(v1) * np.linalg.norm(v2)
    # 避免除以零的情況
    if denom == 0:
        return 180.0  
    
    # 計算餘弦值並剪裁範圍防止浮點數溢位
    cos_thata = np.clip(np.dot(v1, v2) / denom, -1.0, 1.0)

    angle = np.degrees(np.arccos(cos_thata))

    return angle

def get_world_coords(landmark):
    return np.array([landmark.x , landmark.y , landmark.z ]) * 100

def get_canvas_coords(landmark, w, h):
    return int(landmark.x * w), int(landmark.y * h)

def frame(img_bgr, results, w, h):
    left_elbow_angle = 0
    right_elbow_angle = 0
    left_m_distence = 0
    right_m_distence = 0

    try:        
        if not results.pose_world_landmarks:
            return 180, 180, 1, 1, 1
            
        world_landmarks = results.pose_world_landmarks.landmark
        canvas_landmarks = results.pose_landmarks.landmark

        p11_w = get_world_coords(world_landmarks[11]) #左肩
        p12_w = get_world_coords(world_landmarks[12]) #右肩
        shoulder_width = np.linalg.norm(p11_w - p12_w)

        p9_w = get_world_coords(world_landmarks[9])   #嘴左角
        p10_w = get_world_coords(world_landmarks[10]) #嘴右角
        ''' ---------- 3D 計算嘴角二點中心點 ---------- '''
        mouth_center_w = (p9_w + p10_w) / 2
        
        ''' ---------- 2D 顯示嘴角二點中心點 ---------- '''
        p9_2d = get_canvas_coords(canvas_landmarks[9], w, h)
        p10_2d = get_canvas_coords(canvas_landmarks[10], w, h)
        mouth_center_c = (int((p9_2d[0] + p10_2d[0]) / 2), int((p9_2d[1] + p10_2d[1]) / 2))
        cv2.circle(img_bgr, mouth_center_c, 8, (0, 255, 0), -1)


        p11_w = get_world_coords(world_landmarks[11]) #左肩
        p13_w = get_world_coords(world_landmarks[13]) #左肘
        p15_w = get_world_coords(world_landmarks[15]) #左腕
        p17_w = get_world_coords(world_landmarks[17]) #左小指
        p19_w = get_world_coords(world_landmarks[19]) #左食指
        p21_w = get_world_coords(world_landmarks[21]) #左姆指

        ''' ---------- 3D 計算左手肘角度 ---------- '''
        left_elbow_angle = calculate_angle_3d(p11_w, p13_w, p15_w)

        ''' ---------- 2D 顯示, 3D 數值. 左手肘角度 ---------- '''
        cv2.putText(img_bgr, f"Left Elbow: {int(left_elbow_angle)} deg",
                    (660, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)

        """ ---------- 3D 左手部中心點 與 嘴角二點中心點的距離 ---------- """
        left_hand_center_w = np.mean([p15_w, p17_w, p19_w, p21_w], axis=0)
        left_m_distence = np.linalg.norm(left_hand_center_w - mouth_center_w)

        ''' ---------- 2D 顯示, 3D 數值. 嘴角二點中心點的距離 ---------- '''
        cv2.putText(img_bgr, f"L- M Dist: {left_m_distence:.0f} cm", (660, 100), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)


        p12_w = get_world_coords(world_landmarks[12]) #右肩
        p14_w = get_world_coords(world_landmarks[14]) #右肘
        p16_w = get_world_coords(world_landmarks[16]) #右腕
        p18_w = get_world_coords(world_landmarks[18]) #右小指
        p20_w = get_world_coords(world_landmarks[20]) #右食指
        p22_w = get_world_coords(world_landmarks[22]) #右姆指

        ''' ---------- 3D 計算右手肘角度 ---------- '''
        right_elbow_angle = calculate_angle_3d(p12_w, p14_w, p16_w)
        
        ''' ---------- 2D 顯示, 3D 數值. 右手肘角度 ---------- '''
        cv2.putText(img_bgr, f"Right Elbow: {int(right_elbow_angle)} deg",
                    (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

        """ ---------- 3D 右手部中心點 與 嘴角二點中心點的距離 ---------- """
        right_hand_center_w = np.mean([p16_w, p18_w, p20_w, p22_w], axis=0)
        right_m_distence = np.linalg.norm(right_hand_center_w - mouth_center_w)
        
        ''' ---------- 2D 顯示, 3D 數值. 嘴角二點中心點的距離 ---------- '''
        cv2.putText(img_bgr, f"R- M Dist: {right_m_distence:.0f} cm", (30, 100), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

        ''' ---------- 畫 Pose's landmarks ---------- '''
        mp_drawing.draw_landmarks(
            img_bgr,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=4),
            mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
        )

        return left_elbow_angle, right_elbow_angle, left_m_distence, right_m_distence, shoulder_width

    except Exception as e:
        print("Error:", e)
        return 180, 180, 1.0, 1.0, 1.0