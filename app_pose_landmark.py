from flask import Flask, render_template, Response
import cv2
import mediapipe as mp
import numpy as np
import time as t

app = Flask(__name__)

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        smooth_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5)

def show_landmarks(img, pose):
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_rgb.flags.writeable = False            
    results = pose.process(img_rgb)
    img.flags.writeable = True
    
    height, width, _ = img.shape
    landmarks = []
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(
            img, 
            results.pose_landmarks, 
            mp_pose.POSE_CONNECTIONS,
            # mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
            # mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
        )
        for landmark in results.pose_landmarks.landmark:
            landmarks.append(np.array([landmark.x * width, landmark.y * height, landmark.z]))
    return img, landmarks

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

def detect_pose(img, landmarks, w, h):

    left_elbow_angle = 0
    right_elbow_angle = 0
    left_distence = 0
    right_distence = 0


    ''' ---------- 嘴部landmark 嘴左角[9]-嘴右角[10] ---------- '''
    p9 = (landmarks[9], w, h)   #嘴左角
    p10 = (landmarks[10], w, h) #嘴右角

    ''' ---------- 計算嘴角二點中心點 ---------- '''
    mouth_center_point = (p9 + p10) / 2

    ''' ---------- 顯示嘴角二點中心點 ---------- '''
    cv2.circle(img, (int(mouth_center_point[0]), int(mouth_center_point[1])), 8, (0, 255, 0), -1)


    ''' ---------- 左手部landmark 左肩[11]-左肘[13]-左腕[15], 左腕[15]-左小指[17]-左食指[19]-左姆指[21] ---------- '''
    p11 = (landmarks[11], w, h) #左肩
    p13 = (landmarks[13], w, h) #左肘
    p15 = (landmarks[15], w, h) #左腕
    p17 = (landmarks[17], w, h) #左小指
    p19 = (landmarks[19], w, h) #左食指
    p21 = (landmarks[21], w, h) #左姆指

    ''' ---------- 計算左手肘角度 ---------- '''
    left_elbow_angle = calculate_angle(p11, p13, p15)

    ''' ---------- 顯示左手肘角度 ---------- '''
    cv2.putText(
        img, f"Left Elbow: {int(left_elbow_angle)} deg",
        (660, 50), cv2.FONT_HERSHEY_SIMPLEX,
        0.9, (0, 255, 255), 2
    )
    
    ''' ---------- 著色>左手部面積 BGR ---------- '''
    points_int = np.array([p15, p17, p19, p21], dtype=np.int32).reshape((-1, 1, 2))
    cv2.fillPoly(img, [points_int], (255, 200, 0))

    ''' ---------- 計算左手部四點的平均作為中心點 ---------- '''
    left_hand_points = np.array([p15, p17, p19, p21])
    left_hand_center = np.mean(left_hand_points, axis=0)

    ''' ---------- 著色>左手部中心點 ---------- '''
    cv2.circle(img, (int(left_hand_center[0]), int(left_hand_center[1])), 8, (0, 255, 255), -1)

    """ ------ 左手部中心點到嘴角二點中心點的距離 ------ """
    left_distence = np.linalg.norm(left_hand_center - mouth_center_point)
    cv2.putText(img, f"Left Hand to ",
                (660, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    cv2.putText(img, f"Mouth distence: {int(left_distence)} px",
                (660, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)


    ''' ---------- 右手部landmark 右肩[12]-右肘[14]-右腕[16], 右腕[16]-右小指[18]-右食指[20]-右姆指[22] ---------- '''
    p12 = (landmarks[12], w, h) #右肩
    p14 = (landmarks[14], w, h) #右肘
    p16 = (landmarks[16], w, h) #右腕
    p18 = (landmarks[18], w, h) #右小指
    p20 = (landmarks[20], w, h) #右食指
    p22 = (landmarks[22], w, h) #右姆指

    ''' ---------- 計算右手肘角度 ---------- '''
    right_elbow_angle = calculate_angle(p12, p14, p16)

    ''' ---------- 顯示右手肘角度 ---------- '''
    cv2.putText(
        img, f"Right Elbow: {int(right_elbow_angle)} deg",
        (30, 50), cv2.FONT_HERSHEY_SIMPLEX,
        0.9, (255, 0, 255), 2
    )

    ''' ---------- 著色>右手部面積 BGR ---------- '''
    points_int = np.array([p16, p18, p20, p22], dtype=np.int32).reshape((-1, 1, 2))
    cv2.fillPoly(img, [points_int], (255, 0, 0))

    ''' ---------- 計算右手部四點的平均作為中心點 ---------- '''
    right_hand_points = np.array([p16, p18, p20, p22])
    right_hand_center = np.mean(right_hand_points, axis=0)
    
    ''' ---------- 著色>右手部中心點 ---------- '''
    cv2.circle(img, (int(right_hand_center[0]), int(right_hand_center[1])), 8, (125, 0, 255), -1)

    """ ------ 右手部中心點到嘴角二點中心點的距離 ------ """
    right_distence = np.linalg.norm(right_hand_center - mouth_center_point)
    cv2.putText(img, f"Right Hand to ",
                (30, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 255), 2)
    cv2.putText(img, f"Mouth distence: {int(right_distence)} px",
                (30, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 255), 2)

    return left_elbow_angle, right_elbow_angle, left_distence, right_distence

cap = None
def cap_real_time():
    global cap
        
    if cap is None or not cap.isOpened():
        cap = cv2.VideoCapture(0)
        w, h = 1000, 750

    while cap.isOpened():
        ret, img = cap.read()
        if not ret:
            break

        img = cv2.flip(img, 1)

        # ''' ---------- 取得並顯示目前時間 ---------- '''
        current_time = t.localtime()
        format_time = t.strftime("%Y-%m-%d %A %H:%M:%S", current_time)
        now = format_time
        cv2.putText(img, f"{now}", (115, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

        # ''' ---------- 顯示pose landmarks ---------- '''
        img, landmarks = show_landmarks(img, pose)

        # ''' ---------- 螢幕顯示動作姿勢 ---------- '''
        if landmarks:
            img = detect_pose(img, landmarks, w, h)

        ret, buffer = cv2.imencode('.jpg', img)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/start', methods=['POST'])
def start():
    return render_template('index.html')

@app.route('/stop',methods=['POST'])
def stop():
    global cap
    if cap is not None and 'cap' in globals():
        if cap.isOpened():
            cap.release()
    return render_template('stop.html')

@app.route('/cap_in_html')
def cap_in_html():
    return Response(cap_real_time(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=7337)