from flask import Flask, render_template, Response
import cv2
import time as t
import DrawUtil

app = Flask(__name__)

def cap_real_time():
    # dont need to change values here
    cap = cv2.VideoCapture(0)
    cap.set(3, 1024)
    cap.set(4, 768)
    current_eat_state = "ready"
    count = 0

    eating_frame_count = 0
    ACTION_THRESHOLD = 2 #fps
    last_count_time = 0
    COOLDOWN_SECONDS = 2  # seconds
    # ------------------------------
    while cap.isOpened():
        ok, frame = cap.read()
        if not ok:
            continue
        
        frame_height, frame_width, _ = frame.shape
        frame = cv2.resize(frame, (int(frame_width * (650 / frame_height)), 650))

        frame, landmarks = DrawUtil.show_landmarks(frame, DrawUtil.pose)

        # 偵測吃藥動作
        if landmarks:
            frame, left_elbow_angle, right_elbow_angle, l_2_m_distance, r_2_m_distance = DrawUtil.detectPose(frame, landmarks)
            
            # 計算吃藥次數
            # 判斷是否處於吃藥姿勢
            is_eating_pose = (left_elbow_angle <= 60 and l_2_m_distance <= 100) or (right_elbow_angle <= 60 and r_2_m_distance <= 100)
            # 判斷是否處於結束姿勢
            is_resting_pose = (left_elbow_angle >= 90) or (right_elbow_angle >= 90)
            # 重置計數
            if count == 2:
                count = 0
            # 抓時間
            current_time = t.time()
            if is_eating_pose:
                eating_frame_count += 1
                if eating_frame_count >= ACTION_THRESHOLD and(current_time - last_count_time) > COOLDOWN_SECONDS:
                    count += 1
                    current_eat_state = "Eating"
                    last_count_time = current_time
                    eating_frame_count = 0
            else:
                eating_frame_count = 0
            if is_resting_pose:
                current_eat_state = "Detecting"    

            cv2.putText(frame, f'Current State: {current_eat_state}', (30, 200), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 255), 3)
            cv2.putText(frame, f'Count: {count}', (30, 250), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 255), 3)

        else:
            frame = cv2.flip(frame, 1)
            cv2.putText(frame, "No Pose Detected", (80, 400), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 255), 7)
        ''' -------- if wanna put other thing, must after this line -------- '''            

        # 顯示目前時間
        current_time = t.localtime()
        format_time = t.strftime("%Y-%m-%d %A %H:%M:%S", current_time)
        now = format_time
        cv2.putText(frame, f"{now}", (80, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 0), 3)  

        ret, jpeg = cv2.imencode('.jpg', frame)
        frame = jpeg.tobytes()
        yield (b'--frame\r\n'
            b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
    cap.release()
    cv2.destroyAllWindows()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/cap_in_html')
def cap_in_html():
    return Response(cap_real_time(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=7337)
