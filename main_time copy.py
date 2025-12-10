import cv2
import mediapipe as mp
import time as t
import DrawUtil
import pygame

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

w, h = 1000, 750
cap = cv2.VideoCapture(0)

detect, alarm = 5, 3
detect_start_time = None
alarm_start_time = None
wait_time = None
Delay_second = 1.0
current_timer_state = "DETECT"

alarm_sound_file = "alarm.mp3"
pygame.mixer.init()
try:
    pygame.mixer.music.load(alarm_sound_file)
    print("音樂檔案載入成功")
    sound_loaded = True
except pygame.error as e:
    print(f"載入音樂檔案失敗，請檢查檔案路徑和格式: {e}")
    sound_loaded = False

is_alarm_sound_playing = False

with mp_pose.Pose(min_detection_confidence=0.5,min_tracking_confidence=0.5) as pose:

    while cap.isOpened():
        ret, frame = cap.read()
        img_bgr = cv2.resize(frame, (w, h))

        if not ret:
            print("Ignoring empty camera frame.")
            break

        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        results = pose.process(img_rgb)

        (img_bgr, 
         left_elbow_angle, right_elbow_angle, 
         left_distence, right_distence) = DrawUtil.frame(img_bgr, results, w, h)

        current_time = t.localtime()
        format_time = t.strftime("%Y-%m-%d %A %H:%M:%S", current_time)
        now = format_time

        ''' ---------- 判斷有無吃藥動作 ---------- '''
        is_eating_medicine = False
        
        if left_elbow_angle < 50 and left_distence < 150 :
            is_eating_medicine = True
            cv2.putText(img_bgr, "Eat Meduicine", (50, 350), cv2.FONT_HERSHEY_SIMPLEX, 4, (255, 0, 0), 10)
        elif right_elbow_angle < 50 and right_distence < 150 :
            is_eating_medicine = True
            cv2.putText(img_bgr, "Eat Meduicine", (50, 350), cv2.FONT_HERSHEY_SIMPLEX, 4, (255, 0, 0), 10)

        ''' ---------- 計時 ---------- '''
        current_time = t.time()

        if is_eating_medicine:
            if sound_loaded and is_alarm_sound_playing:
                pygame.mixer.music.stop()
                is_alarm_sound_playing = False
            detect_start_time = None
            alarm_start_time = None
            wait_time = None
            current_timer_state = "DETECT"

            ''' ---------- 偵測計時 ---------- '''
        elif current_timer_state == "DETECT":
            if sound_loaded and is_alarm_sound_playing:
                pygame.mixer.music.stop()
                is_alarm_sound_playing = False

            if detect_start_time is None:
                detect_start_time = current_time

            elapsed_time = current_time - detect_start_time
            remaining_time = max(0, detect - int(elapsed_time))

            min, secs = divmod(remaining_time, 60)
            detect_timer = '{:02d}:{:02d}'.format(min, secs)
            cv2.putText(img_bgr, "Detect:", (390, 70), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 4)
            cv2.putText(img_bgr, detect_timer, (430, 130), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 4)

            if remaining_time == 0:
                print("Detect Timer Finished", end='\n\n')
                detect_start_time = None
                wait_time = None
                wait_time = current_time
                current_timer_state = "WAIT_TO_ALARM"

            ''' ---------- 等待1s, 警示計時 ---------- '''
        elif current_timer_state == "WAIT_TO_ALARM":
            if sound_loaded and is_alarm_sound_playing:
                pygame.mixer.music.stop()
                is_alarm_sound_playing = False

            cv2.putText(img_bgr, "Detect:", (390, 70), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 4)
            cv2.putText(img_bgr, detect_timer, (430, 130), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 4)
            elapsed_wait = current_time - wait_time
            if elapsed_wait >= Delay_second:
                print("Wait for ALARM Timer", end='\n\n')
                wait_time = None
                alarm_start_time = None
                current_timer_state = "ALARM"

            ''' ---------- 警示計時 ---------- '''
        elif current_timer_state == "ALARM":
            if sound_loaded and not is_alarm_sound_playing:
                pygame.mixer.music.play(-1)
                is_alarm_sound_playing = True

            if alarm_start_time is None:
                alarm_start_time = current_time

            elapsed_time = current_time - alarm_start_time
            remaining_time = max(0, alarm - int(elapsed_time))

            min, secs = divmod(remaining_time, 60)
            alarm_timer = '{:02d}:{:02d}'.format(min, secs)
            cv2.putText(img_bgr, "ALARM:", (423, 70), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 4)
            cv2.putText(img_bgr, alarm_timer, (430, 130), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 4)

            if remaining_time == 0:
                print("ALARM Timer Finished", end='\n\n')
                alarm_start_time = None
                wait_time = None
                wait_time = current_time
                current_timer_state = "WAIT_TO_DETECT"

            ''' ---------- 等待1s, 偵測計時 ---------- '''
        elif current_timer_state == "WAIT_TO_DETECT":
            if sound_loaded and is_alarm_sound_playing:
                pygame.mixer.music.stop()
                is_alarm_sound_playing = False

            cv2.putText(img_bgr, "ALARM:", (423, 70), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 4)
            cv2.putText(img_bgr, alarm_timer, (430, 130), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 4)
            elapsed_wait = current_time - wait_time
            if elapsed_wait >= Delay_second:
                print("Wait for Detect Timer", end='\n\n')
                wait_time = None
                detect_start_time = None
                current_timer_state = "DETECT"

        ''' ---------- 顯示目前時間 ---------- '''
        cv2.putText(img_bgr, f"{now}", (120, 200), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 0), 5)

        ''' ---------- 顯示畫面 ---------- '''
        cv2.imshow("OOOOKKKK", img_bgr)

        ''' ---------- 按 q 離開 ---------- '''
        if cv2.waitKey(10) == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()