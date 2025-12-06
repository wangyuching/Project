import cv2
import mediapipe as mp
import numpy as np

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

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

''' ------ 座標轉換 (將正規化座標 [0, 1] 轉換為實際像素座標 [0*w, 1*h]) ------ '''
def get_coords(landmark, w, h):
    return np.array([landmark.x * w, landmark.y * h])

def frame(img_bgr, results, w, h):

    left_elbow_angle = 0
    right_elbow_angle = 0
    left_distence = 0
    right_distence = 0

    try:        
        if not results.pose_landmarks:
            return img_bgr, left_elbow_angle, right_elbow_angle, left_distence, right_distence
            
        landmarks = results.pose_landmarks.landmark

        ''' ---------- 左手部landmark 左肩[11]-左肘[13]-左腕[15], 左腕[15]-左小指[17]-左食指[19]-左姆指[21] ---------- '''
        p11 = get_coords(landmarks[11], w, h) #左肩
        p13 = get_coords(landmarks[13], w, h) #左肘
        p15 = get_coords(landmarks[15], w, h) #左腕
        p17 = get_coords(landmarks[17], w, h) #左小指
        p19 = get_coords(landmarks[19], w, h) #左食指
        p21 = get_coords(landmarks[21], w, h) #左姆指

        ''' ---------- 計算左手肘角度 ---------- '''
        left_elbow_angle = calculate_angle(p11, p13, p15)

        ''' ---------- 顯示左手肘角度 ---------- '''
        cv2.putText(
            img_bgr, f"Left Elbow: {int(left_elbow_angle)} deg",
            (670, 50), cv2.FONT_HERSHEY_SIMPLEX,
            1, (0, 255, 255), 2
        )
        
        ''' ---------- 著色>左手部面積 BGR ---------- '''
        points_int = np.array([p15, p17, p19, p21], dtype=np.int32).reshape((-1, 1, 2))
        cv2.fillPoly(img_bgr, [points_int], (255, 200, 0))

        ''' ---------- 計算左手部四點的平均作為中心點 ---------- '''
        left_hand_points = np.array([p15, p17, p19, p21])
        left_hand_center = np.mean(left_hand_points, axis=0)

        ''' ---------- 著色>左手部中心點 ---------- '''
        cv2.circle(img_bgr, (int(left_hand_center[0]), int(left_hand_center[1])), 8, (0, 255, 255), -1)


        ''' ---------- 右手部landmark 右肩[12]-右肘[14]-右腕[16], 右腕[16]-右小指[18]-右食指[20]-右姆指[22] ---------- '''
        p12 = get_coords(landmarks[12], w, h) #右肩
        p14 = get_coords(landmarks[14], w, h) #右肘
        p16 = get_coords(landmarks[16], w, h) #右腕
        p18 = get_coords(landmarks[18], w, h) #右小指
        p20 = get_coords(landmarks[20], w, h) #右食指
        p22 = get_coords(landmarks[22], w, h) #右姆指

        ''' ---------- 計算右手肘角度 ---------- '''
        right_elbow_angle = calculate_angle(p12, p14, p16)

        ''' ---------- 顯示右手肘角度 ---------- '''
        cv2.putText(
            img_bgr, f"Right Elbow: {int(right_elbow_angle)} deg",
            (50, 50), cv2.FONT_HERSHEY_SIMPLEX,
            1, (0, 0, 255), 2
        )

        ''' ---------- 著色>右手部面積 BGR ---------- '''
        points_int = np.array([p16, p18, p20, p22], dtype=np.int32).reshape((-1, 1, 2))
        cv2.fillPoly(img_bgr, [points_int], (255, 0, 0))

        ''' ---------- 計算右手部四點的平均作為中心點 ---------- '''
        right_hand_points = np.array([p16, p18, p20, p22])
        right_hand_center = np.mean(right_hand_points, axis=0)
        
        ''' ---------- 著色>右手部中心點 ---------- '''
        cv2.circle(img_bgr, (int(right_hand_center[0]), int(right_hand_center[1])), 8, (125, 0, 255), -1)


        ''' ---------- 嘴部landmark 嘴左角[9]-嘴右角[10] ---------- '''
        p9 = get_coords(landmarks[9], w, h)   #嘴左角
        p10 = get_coords(landmarks[10], w, h) #嘴右角

        ''' ---------- 計算嘴角二點中心點 ---------- '''
        mouth_center_point = (p9 + p10) / 2

        ''' ---------- 顯示嘴角二點中心點 ---------- '''
        cv2.circle(img_bgr, (int(mouth_center_point[0]), int(mouth_center_point[1])), 8, (0, 255, 0), -1)


        """ ------ 左手部中心點到嘴角二點中心點的距離 ------ """
        left_distence = np.linalg.norm(left_hand_center - mouth_center_point)
        cv2.putText(img_bgr, f"Left Hand to Mouth distence: {int(left_distence)} px",
                    (670, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        """ ------ 右手部中心點到嘴角二點中心點的距離 ------ """
        right_distence = np.linalg.norm(right_hand_center - mouth_center_point)
        cv2.putText(img_bgr, f"Right Hand to Mouth distence: {int(right_distence)} px",
                    (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (125, 0, 255), 2)
        
        ''' ---------- 畫 Pose's landmarks ---------- '''
        mp_drawing.draw_landmarks(
            img_bgr,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=4),
            mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
        )

        return img_bgr, left_elbow_angle, right_elbow_angle, left_distence, right_distence

    except Exception as e:
        print("Error:", e)
        return img_bgr, left_elbow_angle, right_elbow_angle, left_distence, right_distence
