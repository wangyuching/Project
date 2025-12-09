import cv2
import mediapipe as mp
import time as t
import DrawUtil
from datetime import datetime as dt

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

w, h = 1000, 750
detect, alert = 5, 3
cap = cv2.VideoCapture(0)

count_eating_medicine = 0

detect_start_time = None
alert_start_time = None
current_timer_state = "DETECT"

with mp_pose.Pose(min_detection_confidence=0.5,min_tracking_confidence=0.5) as pose:

    while cap.isOpened():
        ret, frame = cap.read()
        img_bgr = cv2.resize(frame, (w, h))

        if not ret:
            print("Ignoring empty camera frame.")
            break

        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        results = pose.process(img_rgb)

        (DrawUtil_img_bgr, 
         left_elbow_angle, right_elbow_angle, 
         left_distence, right_distence) = DrawUtil.frame(img_bgr, results, w, h)

        now = dt.now().strftime('%Y-%m-%d %A %H:%M:%S')

        ''' ---------- 判斷吃藥動作 ---------- '''
        is_eating_medicine = False
        
        if left_elbow_angle < 50 and left_distence < 150 :
            count_eating_medicine += 1
            is_eating_medicine = True
            cv2.putText(DrawUtil_img_bgr, "Eat Meduicine", (50, 350), cv2.FONT_HERSHEY_SIMPLEX, 5, (255, 0, 0), 15)
        elif right_elbow_angle < 50 and right_distence < 150 :
            count_eating_medicine += 1
            is_eating_medicine = True
            cv2.putText(DrawUtil_img_bgr, "Eat Meduicine", (50, 350), cv2.FONT_HERSHEY_SIMPLEX, 5, (255, 0, 0), 15)
        else:
            pass

        cv2.putText(DrawUtil_img_bgr, f"Count: {count_eating_medicine}", (50, 350), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 4)
        
        try:
            ''' ---------- 計時 ---------- '''
            current_time = t.time()
            if current_timer_state == "DETECT":
                if detect_start_time is None:
                    detect_start_time = current_time

                elapsed_time = current_time - detect_start_time
                remaining_time = max(0, detect - round(elapsed_time))

                min, secs = divmod(remaining_time, 60)
                detect_timer = '{:02d}:{:02d}'.format(min, secs)
                cv2.putText(DrawUtil_img_bgr, "Detect:", (390, 70), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 4)
                cv2.putText(DrawUtil_img_bgr, detect_timer, (430, 130), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 4)

                if is_eating_medicine and count_eating_medicine >= 2:
                    current_timer_state = "DETECT"
                    detect_start_time = None
                    count_eating_medicine = 0
                elif remaining_time <= 0 and not is_eating_medicine:
                    print("END 1 - Detect Timer Finished", end='\n\n')
                    current_timer_state = "ALERT"
                    alert_start_time = None
                    count_eating_medicine = 0

            elif current_timer_state == "ALERT":
                if alert_start_time is None:
                    alert_start_time = current_time

                elapsed_time = current_time - alert_start_time
                remaining_time = max(0, alert - round(elapsed_time))

                min, secs = divmod(remaining_time, 60)
                alert_timer = '{:02d}:{:02d}'.format(min, secs)
                cv2.putText(DrawUtil_img_bgr, "Alert:", (423, 70), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 4)
                cv2.putText(DrawUtil_img_bgr, alert_timer, (430, 130), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 4)

                if is_eating_medicine and count_eating_medicine >= 2:
                    current_timer_state = "DETECT"
                    detect_start_time = None
                    count_eating_medicine = 0
                elif remaining_time <= 0 and not is_eating_medicine:
                    print("END 2 - Alert Timer Finished", end='\n\n')
                    current_timer_state = "DETECT"
                    detect_start_time = None
                    count_eating_medicine = 0

            ''' ---------- 顯示目前時間 ---------- '''
            cv2.putText(DrawUtil_img_bgr, f"{now}", (120, 200), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 0), 5)

            ''' ---------- 顯示畫面 ---------- '''
            cv2.imshow("OOOOKKKK", DrawUtil_img_bgr)

            ''' ---------- 按 q 離開 ---------- '''
            if cv2.waitKey(10) == ord('q'):
                break

        except Exception as e:
            print("Error:", e)
            break

cap.release()
cv2.destroyAllWindows()