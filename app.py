from flask import Flask, render_template, Response, jsonify
import cv2
import time as t
import DrawUtil
import mysql.connector
from mysql.connector import pooling
import threading

app = Flask(__name__)

from mysql.connector import pooling

db_config = {
    "host": "127.0.0.1",
    "port": 3306,
    "user": "root",
    "password": "DataBase",
    "database": "project"
}

try:
    connection_pool = pooling.MySQLConnectionPool(
        pool_name="med_pool",
        pool_size=5,
        **db_config
    )
    print("Database Connection Pool 建立成功")
except mysql.connector.Error as err:
    print(f"建立連線池失敗: {err}")
    connection_pool = None

def save_to_db( log_time, state, auto_finish):
    if not connection_pool:
        print("連線池未就緒，無法寫入")
        return
    connect = None
    try:
        connect = connection_pool.get_connection()
        cursor = connect.cursor()
        sql = "INSERT INTO `take_medicine` (time, state, auto_finish) VALUES (%s, %s, %s)"
        cursor.execute(sql, (log_time, state, auto_finish))
        connect.commit()
    except Exception as e:
        print(f"背景寫入失敗: {e}")
    finally:
        if connect and connect.is_connected():
            cursor.close()
            connect.close()

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
    try:
        while cap.isOpened():
            ok, frame = cap.read()
            if not ok:
                break
            
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
                    if eating_frame_count > ACTION_THRESHOLD and (current_time - last_count_time) > COOLDOWN_SECONDS:
                        count += 1
                        current_eat_state = "Eating"
                        last_count_time = current_time
                        eating_frame_count = 0
                        # 記錄時間
                        log_time = t.strftime("%Y-%m-%d %H:%M:%S", t.localtime(current_time))

                        state = "Start" if count == 1 else "Finish"
                        thread = threading.Thread(target=save_to_db, args=(log_time, state, ""))
                        thread.start()

                        if count == 1:
                            first_eat_time = current_time
                        elif count == 2:
                            count = 0
                            first_eat_time = 0
                else:
                    eating_frame_count = 0
                    current_eat_state = "Detecting" 
            else:
                frame = cv2.flip(frame, 1)
                cv2.putText(frame, "No Pose Detected", (80, 400), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 255), 7)

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
    return render_template('home.html', title='Home')

@app.route('/api/get_history')
def get_history():
    connect = None
    history = []
    try:
        connect = connection_pool.get_connection()
        cursor = connect.cursor(dictionary=True)
        sql = """
            SELECT 
                DATE_FORMAT(time, '%Y-%m-%d %H:%i:%S') as time,
                state, 
                auto_finish 
            FROM `take_medicine`;
        """
        cursor.execute(sql)
        history = cursor.fetchall()
    except mysql.connector.Error as err:
        print(f"Error: {err}")
    finally:
        if connect and connect.is_connected():
            cursor.close()
            connect.close()
    return jsonify(history)

@app.route('/cap_in_html')
def cap_in_html():
    return Response(cap_real_time(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True, host='127.0.0.1', port=3377)
