from flask import Flask, render_template, Response
import cv2
import mediapipe as mp

app = Flask(__name__)

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

cap = None

def cap_real_time():
    global cap

    with mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        smooth_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as pose:
        
        if cap is None or not cap.isOpened():
            cap = cv2.VideoCapture(0)
        
        while cap.isOpened():
            ret, img = cap.read()
            if not ret:
                break
            
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_rgb.flags.writeable = False
            
            results = pose.process(img_rgb)
            
            img.flags.writeable = True
            if results.pose_landmarks:
                mp_drawing.draw_landmarks(
                    img, 
                    results.pose_landmarks, 
                    mp_pose.POSE_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2), # 關節點
                    mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2)  # 連接線
                )

            ret, buffer = cv2.imencode('.jpg', img)
            if not ret:
                continue
                
            frame = buffer.tobytes()
            
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        
        if cap:
            cap.release()

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
    # 串流回應
    return Response(cap_real_time(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=7337)