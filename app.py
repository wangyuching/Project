from flask import Flask, render_template, Response, jsonify
from flask_sqlalchemy import SQLAlchemy
import cv2
import time as t
import DrawUtil
import threading
import base64

app = Flask(__name__)

app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql+pymysql://root:DataBase@127.0.0.1:3306/project'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

class take_medicine(db.Model):
    __tablename__ = 'take_medicine'
    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    state = db.Column(db.String(10))
    time = db.Column(db.DateTime)
    auto_finish = db.Column(db.String(50))
    pose_img1 = db.Column(db.LargeBinary(length=(2**16)-1))
    pose_img2 = db.Column(db.LargeBinary(length=(2**16)-1))
    pose_img3 = db.Column(db.LargeBinary(length=(2**16)-1))
    pose_img4 = db.Column(db.LargeBinary(length=(2**16)-1))

with app.app_context():
    db.create_all()

lock = threading.Lock()
latest_frame = None
global_data = {
    "count": 0,
    "current_eat_state": "Detecting",
    "first_eat_time": 0
}
last_active_time = 0

def save_to_db(state, log_time, auto_finish, frames=None):
    with app.app_context():
        try:
            blob_img = [None, None, None, None]
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 50]
            if frames:
                for i, f in enumerate(frames[:4]):
                    if f is not None:
                        _, buffer = cv2.imencode('.jpg', f, encode_param)
                        img_bytes = buffer.tobytes()
                        if len(img_bytes) < (2**16)-1:
                            blob_img[i] = img_bytes
                        else:
                            print(f"Image {i+1} exceeds size limit, not saved.")

            new_record = take_medicine(
                state=state, 
                time=log_time, 
                auto_finish=auto_finish,
                pose_img1=blob_img[0],
                pose_img2=blob_img[1],
                pose_img3=blob_img[2],
                pose_img4=blob_img[3]
                )
            db.session.add(new_record)
            db.session.commit()
            print(f"Record saved: {state}, {log_time}, {auto_finish}, poseImg{len([b for b in blob_img if b])}")
        except Exception as e:
            db.session.rollback()
            print(f"Error saving record: {e}")

def cap_worker():
    global latest_frame, global_data, last_active_time
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open camera")
        return

    cap.set(3, 1024)
    cap.set(4, 768)
    current_eat_state = None
    count = 0
    # ------------------------------
    eating_frame_count = 0
    ACTION_THRESHOLD = 3 #fps
    last_count_time = 0
    COOLDOWN_SECONDS = 2  # seconds
    # ------------------------------
    temp_frame = []
    # ------------------------------

    print("背景 Worker 已啟動，等待網頁連線...")

    try:
        while True:
            current_time = t.time()
            if (current_time - last_active_time) < 3:
                if cap is None:
                    print("--- 偵測到網頁活動，啟動攝影機 ---")
                    cap = cv2.VideoCapture(0)
                    cap.set(3, 1024)
                    cap.set(4, 768)

                if cap.isOpened():
                    ok, frame = cap.read()
                    if not ok:
                        break

                    save_frame = frame.copy()
                    frame_height, frame_width, _ = frame.shape

                    processed_frame = cv2.resize(frame, (int(frame_width * (650 / frame_height)), 650))
                    processed_frame, landmarks = DrawUtil.show_landmarks(processed_frame, DrawUtil.pose)

                    current_time = t.time()

                    if landmarks:
                        processed_frame, left_elbow_angle, right_elbow_angle, l_2_m_distance, r_2_m_distance = DrawUtil.detectPose(processed_frame, landmarks)

                        is_eating_pose = (left_elbow_angle <= 60 and l_2_m_distance <= 100) or (right_elbow_angle <= 60 and r_2_m_distance <= 100)                
                        if is_eating_pose:
                            eating_frame_count += 1

                            if len(temp_frame) < 4:
                                temp_frame.append(save_frame)

                            if eating_frame_count > ACTION_THRESHOLD and (current_time - last_count_time) > COOLDOWN_SECONDS:
                                global_data["count"] += 1
                                global_data["current_eat_state"] = "Eating"
                                
                                log_time = t.strftime("%Y-%m-%d %H:%M:%S", t.localtime(current_time))
                                state = "Start" if global_data["count"] == 1 else "Finish"
                                threading.Thread(target=save_to_db, args=(state, log_time, "", list(temp_frame))).start()

                                if global_data["count"] == 1:
                                    global_data["first_eat_time"] = current_time
                                elif global_data["count"] == 2:
                                    global_data["count"] = 0
                                    global_data["first_eat_time"] = 0

                                last_count_time = current_time
                                eating_frame_count = 0
                                temp_frame = []
                        else:
                            eating_frame_count = 0
                            temp_frame = []
                            global_data["current_eat_state"] = "Detecting" 
                    else:
                        frame = cv2.flip(processed_frame, 1)
                        cv2.putText(frame, "No Pose Detected", (80, 400), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 255), 7)
                        temp_frame = []

                    if global_data["count"] == 1 and global_data["first_eat_time"] != 0:
                        if (current_time - global_data["first_eat_time"]) > 30:
                            auto_second_time = t.strftime("%Y-%m-%d %H:%M:%S", t.localtime(global_data["first_eat_time"] + 30))
                            threading.Thread(target=save_to_db, args=("Finish", auto_second_time, "Auto logged after 30 seconds", None)).start()

                            global_data["count"] = 0
                            global_data["first_eat_time"] = 0

                    now = t.strftime("%Y-%m-%d %A %H:%M:%S", t.localtime())
                    cv2.putText(processed_frame, f"{now}", (80, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 0), 3)  

                    cv2.putText(processed_frame, f'Current State: {global_data["current_eat_state"]}', (30, 200), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 255), 3)
                    cv2.putText(processed_frame, f'Count: {global_data["count"]}', (30, 250), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 255), 3)

                    with lock:
                        latest_frame = processed_frame.copy()
            else:
                if cap is not None:
                    print("--- 偵測到無網頁活動，關閉攝影機 ---")

                    if global_data["count"] == 1:
                        auto_cap_close_time = t.strftime("%Y-%m-%d %H:%M:%S", t.localtime(t.time()))
                        threading.Thread(target=save_to_db, args=("Finish", auto_cap_close_time, "Auto logged due to cap close", None)).start()
                        
                    cap.release()
                    cap = None
                    with lock:
                        latest_frame = None
            t.sleep(0.03)

    except Exception as e:
        print(f"Error in video processing: {e}")
    finally:
        if global_data["count"] == 1:
            auto_cap_close_second_time = t.strftime("%Y-%m-%d %H:%M:%S", t.localtime(t.time()))
            threading.Thread(target=save_to_db, args=("Finish", auto_cap_close_second_time, "Auto logged due to cap close", None)).start() 
        cap.release()
        print("Video capture released.")

threading.Thread(target=cap_worker, daemon=True).start()

@app.route('/api/still_alive')
def heartbeat():
    global last_active_time
    last_active_time = t.time()
    return jsonify({"status": "alive"})

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/search')
def search():
    return render_template('search.html')

@app.route('/api/history')
def api_history():
    history = take_medicine.query.all()
    history_list = []
    for record in history:
        def to_base64(img_blob):
            if img_blob:
                return base64.b64encode(img_blob).decode('utf-8')
            return None

        history_list.append({
            'state': record.state,
            'time': record.time.strftime("%Y-%m-%d %H:%M:%S"),
            'auto_finish': record.auto_finish,
            'pose_img1': to_base64(record.pose_img1),
            'pose_img2': to_base64(record.pose_img2),
            'pose_img3': to_base64(record.pose_img3),
            'pose_img4': to_base64(record.pose_img4)
        })
    return jsonify(history_list)

@app.route('/cap_in_html')
def cap_in_html():
    def stream():
        while True:
            with lock:
                if latest_frame is None:
                    continue
                ret, jpeg = cv2.imencode('.jpg', latest_frame)
                if not ret:
                    continue
                frame = jpeg.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')
            t.sleep(0.05)
    return Response(stream(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=False, host='127.0.0.1', port=3377)
