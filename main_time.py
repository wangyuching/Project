import cv2
import mediapipe as mp
import time as t
import DrawUtil
# from datetime import datetime as dt

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

w, h = 1000, 750
detect, alert = 5, 3
detect_start_time = None
alert_start_time = None
wait_time = None
Delay_second = 1.0
current_timer_state = "DETECT"

cap = cv2.VideoCapture(0)

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

        # now =dt.now().strftime('%Y-%m-%d %A %H:%M:%S')
        current_time = t.localtime()
        format_time = t.strftime("%Y-%m-%d %A %H:%M:%S", current_time)
        now = format_time

        ''' ---------- 判斷吃藥動作 ---------- '''
        is_eating_medicine = False
        
        if left_elbow_angle < 50 and left_distence < 150 :
            is_eating_medicine = True
            cv2.putText(DrawUtil_img_bgr, "Eat Meduicine", (50, 350), cv2.FONT_HERSHEY_SIMPLEX, 4, (255, 0, 0), 10)
        elif right_elbow_angle < 50 and right_distence < 150 :
            is_eating_medicine = True
            cv2.putText(DrawUtil_img_bgr, "Eat Meduicine", (50, 350), cv2.FONT_HERSHEY_SIMPLEX, 4, (255, 0, 0), 10)
        
        try:
            ''' ---------- 計時 ---------- '''
            current_time = t.time()

            if is_eating_medicine:
                detect_start_time = None
                alert_start_time = None
                wait_time = None
                current_timer_state = "DETECT"

            elif current_timer_state == "DETECT":
                if detect_start_time is None:
                    detect_start_time = current_time

                elapsed_time = current_time - detect_start_time
                remaining_time = max(0, detect - round(elapsed_time))

                min, secs = divmod(remaining_time, 60)
                detect_timer = '{:02d}:{:02d}'.format(min, secs)
                cv2.putText(DrawUtil_img_bgr, "Detect:", (390, 70), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 4)
                cv2.putText(DrawUtil_img_bgr, detect_timer, (430, 130), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 4)

                if remaining_time == 0:
                    print("END 1 - Detect Timer Finished", end='\n\n')
                    detect_start_time = None
                    wait_time = None
                    wait_time = current_time
                    current_timer_state = "WAITAlert"

            elif current_timer_state == "WAITAlert":
                elapsed_wait = current_time - wait_time
                if elapsed_wait >= Delay_second:
                    print("Wait for Alert Timer", end='\n\n')
                    wait_time = None
                    alert_start_time = None
                    current_timer_state = "ALERT"

            elif current_timer_state == "ALERT":
                if alert_start_time is None:
                    alert_start_time = current_time

                elapsed_time = current_time - alert_start_time
                remaining_time = max(0, alert - round(elapsed_time))

                min, secs = divmod(remaining_time, 60)
                alert_timer = '{:02d}:{:02d}'.format(min, secs)
                cv2.putText(DrawUtil_img_bgr, "Alert:", (423, 70), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 4)
                cv2.putText(DrawUtil_img_bgr, alert_timer, (430, 130), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 4)

                if remaining_time == 0:
                    print("END 2 - Alert Timer Finished", end='\n\n')
                    alert_start_time = None
                    wait_time = None
                    wait_time = current_time
                    current_timer_state = "WAITDetect"

            elif current_timer_state == "WAITDetect":
                elapsed_wait = current_time - wait_time
                if elapsed_wait >= Delay_second:
                    print("Wait for Alert Timer", end='\n\n')
                    wait_time = None
                    detect_start_time = None
                    current_timer_state = "DETECT"

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