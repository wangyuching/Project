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
    count = 0
    current_eat_state = "no_eating"
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
        else:
            cv2.putText(frame, "No Pose Detected", (80, 400), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 255), 7)
        ''' -------- if wanna put other thing, must after this line -------- '''            
        # 計算吃藥次數
        if left_elbow_angle >= 90:
            current_eat_state = "not_eating"
        elif right_elbow_angle >= 90:
            current_eat_state = "not_eating"

        if left_elbow_angle <= 60 and l_2_m_distance <= 100 and current_eat_state == "not_eating":
            current_eat_state = "eating"
            count += 1
        elif right_elbow_angle <= 60 and r_2_m_distance <= 100 and current_eat_state == "not_eating":
            current_eat_state = "eating"
            count += 1

        cv2.putText(frame, f'Current State: {current_eat_state}', (30, 200), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 255), 3)
        cv2.putText(frame, f'Count: {count}', (30, 250), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 255), 3)

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
