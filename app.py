from flask import Flask, render_template, Response, jsonify
from flask_sqlalchemy import SQLAlchemy
import cv2
import time as t
import DrawUtil
import threading
import os

app = Flask(__name__)

app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql+pymysql://root:DataBase@127.0.0.1:3306/project'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)

class take_medicine(db.Model):
    __tablename__ = 'take_medicine'
    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    time = db.Column(db.DateTime)
    state = db.Column(db.String(10))
    auto_finish = db.Column(db.String(50))

with app.app_context():
    db.create_all()

save_path = 'static/poseImgs'
if not os.path.exists(save_path):
    os.makedirs(save_path)

def save_image(frame,state, timestamp):
    for i, f in enumerate(frame):
        filename = f"{state}_{timestamp}_f{i}.jpg"
        filepath = os.path.join(save_path, filename)
        cv2.imwrite(filepath, f)
    print(f"Successfully saved images: {filename}")

def save_to_db( log_time, state, auto_finish):
    with app.app_context():
        try:
            new_record = take_medicine(time=log_time, state=state, auto_finish=auto_finish)
            db.session.add(new_record)
            db.session.commit()
            print(f"Record saved: {log_time}, {state}, {auto_finish}")
        except Exception as e:
            db.session.rollback()
            print(f"Error saving record: {e}")

def cap_real_time():
    # dont need to change values here
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
    first_eat_time = 0
    # ------------------------------
    temp_frame = []
    # ------------------------------
    try:
        while cap.isOpened():
            ok, frame = cap.read()
            if not ok:
                break

            save_frame = frame.copy()
            
            frame_height, frame_width, _ = frame.shape
            frame = cv2.resize(frame, (int(frame_width * (650 / frame_height)), 650))

            frame, landmarks = DrawUtil.show_landmarks(frame, DrawUtil.pose)

            if landmarks:
                frame, left_elbow_angle, right_elbow_angle, l_2_m_distance, r_2_m_distance = DrawUtil.detectPose(frame, landmarks)
                
                # 計算吃藥次數
                # 判斷是否處於吃藥姿勢
                is_eating_pose = (left_elbow_angle <= 60 and l_2_m_distance <= 100) or (right_elbow_angle <= 60 and r_2_m_distance <= 100)
                # 抓時間
                current_time = t.time()
                if is_eating_pose:
                    eating_frame_count += 1
                    temp_frame.append(save_frame)
                    if eating_frame_count > ACTION_THRESHOLD and (current_time - last_count_time) > COOLDOWN_SECONDS:
                        count += 1
                        current_eat_state = "Eating"
                        
                        log_time = t.strftime("%Y-%m-%d %H:%M:%S", t.localtime(current_time))
                        file_time = t.strftime("%Y%m%d_%H%M%S", t.localtime(current_time))
                        state = "Start" if count == 1 else "Finish"
                        if temp_frame:
                            save_thread = threading.Thread(target=save_image, args=(list(temp_frame), state, file_time))
                            save_thread.start()

                        thread = threading.Thread(target=save_to_db, args=(log_time, state, ""))
                        thread.start()

                        if count == 1:
                            first_eat_time = current_time
                        elif count == 2:
                            count = 0
                            first_eat_time = 0

                        last_count_time = current_time
                        eating_frame_count = 0
                        temp_frame = []
                else:
                    eating_frame_count = 0
                    temp_frame = []
                    current_eat_state = "Detecting" 
            else:
                frame = cv2.flip(frame, 1)
                cv2.putText(frame, "No Pose Detected", (80, 400), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 255), 7)
                temp_frame = []

            # 自動記錄第二次吃藥(放在if landmarks外,沒有偵測道動作時也會執行)
            current_time = t.time()
            if count == 1 and first_eat_time != 0:
                if (current_time - first_eat_time) > 30:
                    auto_second_time = t.strftime("%Y-%m-%d %H:%M:%S", t.localtime(first_eat_time + 30))
                    thread = threading.Thread(target=save_to_db, args=(auto_second_time, "Finish", "Auto logged after 30 seconds"))
                    thread.start()

                    count = 0
                    first_eat_time = 0
                    current_eat_state = "Detecting"

            # 顯示目前時間
            current_time = t.localtime()
            format_time = t.strftime("%Y-%m-%d %A %H:%M:%S", current_time)
            now = format_time
            cv2.putText(frame, f"{now}", (80, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 0), 3)  

            # 顯示目前狀態與吃藥次數
            cv2.putText(frame, f'Current State: {current_eat_state}', (30, 200), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 255), 3)
            cv2.putText(frame, f'Count: {count}', (30, 250), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 255), 3)

            ret, jpeg = cv2.imencode('.jpg', frame)
            frame = jpeg.tobytes()
            yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
    except Exception as e:
        print(f"Error in video processing: {e}")
    finally:
        cap.release()
        print("Video capture released.")

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
        history_list.append({
            'time': record.time.strftime("%Y-%m-%d %H:%M:%S"),
            'state': record.state,
            'auto_finish': record.auto_finish
        })
    return jsonify(history_list)

@app.route('/cap_in_html')
def cap_in_html():
    return Response(cap_real_time(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True, host='127.0.0.1', port=3377)
