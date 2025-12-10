import cv2
import mediapipe as mp
import time as t
import DrawUtil
import TimeLogic

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

w, h = 1000, 750
cap = cv2.VideoCapture(0)

detect, alarm = 5, 3
detect_start_time = None
alarm_start_time = None
wait_time = None
current_timer_state = "DETECT"

with mp_pose.Pose(min_detection_confidence=0.5,min_tracking_confidence=0.5) as pose:

    while cap.isOpened():
        ret, frame = cap.read()

        if not ret:
            print("Ignoring empty camera frame.")
            break
        
        img_bgr = cv2.resize(frame, (w, h))
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        results = pose.process(img_rgb)

        ''' ---------- 取得並顯示目前時間 ---------- '''
        current_time = t.localtime()
        format_time = t.strftime("%Y-%m-%d %A %H:%M:%S", current_time)
        now = format_time
        cv2.putText(img_bgr, f"{now}", (120, 200), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 0), 5)

        ''' ---------- 螢幕顯示動作姿勢 ---------- '''
        (left_elbow_angle, right_elbow_angle, 
         left_distence, right_distence) = DrawUtil.frame(img_bgr, results, w, h)

        ''' ---------- 判斷有無吃藥動作 ---------- '''
        is_eating_medicine = False

        if left_elbow_angle < 50 and left_distence < 150 :
            is_eating_medicine = True
            cv2.putText(img_bgr, "Eat Medicine", (50, 350), cv2.FONT_HERSHEY_SIMPLEX, 4, (255, 0, 0), 10)
        elif right_elbow_angle < 50 and right_distence < 150 :
            is_eating_medicine = True
            cv2.putText(img_bgr, "Eat Meduicine", (50, 350), cv2.FONT_HERSHEY_SIMPLEX, 4, (255, 0, 0), 10)

        ''' ---------- 倒數計時器 ---------- '''
        (detect_start_time, alarm_start_time, wait_time, current_timer_state) = TimeLogic.timer(
            img_bgr, is_eating_medicine, 
            detect, alarm, detect_start_time, alarm_start_time, 
            wait_time, current_timer_state)        

        ''' ---------- 顯示畫面 ---------- '''
        cv2.imshow("OOOOKKKK", img_bgr)

        ''' ---------- 按 q 離開 ---------- '''
        if cv2.waitKey(10) == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()