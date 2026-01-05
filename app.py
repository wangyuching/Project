from flask import Flask, render_template, Response
import cv2
import time as t
import DrawUtil
from TimeLogic import timer

app = Flask(__name__)

def cap_real_time():
    # dont need to change values here
    cap = cv2.VideoCapture(0)
    cap.set(3, 1024)
    cap.set(4, 768)
    my_timer = timer()
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
            frame, is_eating_medicine = DrawUtil.detectPose(frame, landmarks)
        else:
            cv2.putText(frame, "No Pose Detected", (80, 400), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 255), 7)
        
        ''' -------- if wanna put other thing, must after this line -------- '''            
        # 目前時間
        current_time = t.localtime()
        format_time = t.strftime("%Y-%m-%d %A %H:%M:%S", current_time)
        now = format_time
        cv2.putText(frame, f"{now}", (80, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 0), 3)

        # ''' ---------- 倒數計時器 ---------- '''
        frame = my_timer.update(frame, is_eating_medicine)   

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
