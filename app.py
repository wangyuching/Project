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
    while cap.isOpened():
        ok, frame = cap.read()
        if not ok:
            continue
        
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
                    log_time = t.strftime("%Y-%m-%d %A %H:%M:%S", t.localtime(current_time))
                    try:
                        with open("eat_log.txt", "a", encoding="utf-8") as f:
                            if count == 1:
                                f.write(f"First eat at: {log_time}\n")
                                first_eat_time = current_time
                            elif count == 2:
                                f.write(f"Second eat at: {log_time}\n\n")
                                count = 0
                                first_eat_time = 0
                    except Exception as e:
                        print(f"Error writing to log file: {e}")
            else:
                eating_frame_count = 0
                current_eat_state = "Detecting" 
        else:
            frame = cv2.flip(frame, 1)
            cv2.putText(frame, "No Pose Detected", (80, 400), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 255), 7)

        # 自動記錄第二次吃藥
        current_time = t.time()
        if count == 1 and first_eat_time != 0:
            if (current_time - first_eat_time) > 30:
                second_eat_time = t.strftime("%Y-%m-%d %A %H:%M:%S", t.localtime(first_eat_time + 30))
                try:
                    with open("eat_log.txt", "a", encoding="utf-8") as f:
                        f.write(f"Second eat at: {second_eat_time} (Auto logged after 30 seconds)\n\n")
                except Exception as e:
                    print(f"Error writing to log file: {e}")

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
    cap.release()
    cv2.destroyAllWindows()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/cap_in_html')
def cap_in_html():
    return Response(cap_real_time(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/get_txt')
def get_txt():
    try:
        with open("eat_log.txt", "r", encoding="utf-8") as f:
            content = f.read()
        return content
    except FileNotFoundError:
        return "Log file not found."
    
@app.route('/clear_txt', methods=['POST'])
def clear_txt():
    try:
        with open("eat_log.txt", "w", encoding="utf-8") as f:
            f.write("")
        return "Log file cleared."
    except Exception as e:
        return f"Error clearing log file: {e}"

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=7337)
