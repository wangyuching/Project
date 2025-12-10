import cv2
import time as t
import pygame

Delay_second = 1.0

pygame.mixer.init()
try:
    pygame.mixer.music.load("alarm3.mp3")
except Exception as e:
    print("Error loading alarm sound:", e)

def timer(img_bgr, is_eating_medicine, 
           detect, alarm, detect_start_time, alarm_start_time, 
           wait_time, current_timer_state):
    try:
        ''' ---------- 取得目前時間戳 ---------- '''
        current_time = t.time()

        ''' ---------- 狀態 0: 判斷有無吃藥動作 ---------- '''
        if is_eating_medicine:
            pygame.mixer.music.stop()
            detect_start_time = None
            alarm_start_time = None
            wait_time = None
            current_timer_state = "DETECT"

            ''' ---------- 狀態 1: 偵測動作倒計時: ---------- '''
        elif current_timer_state == "DETECT":
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

            ''' ---------- 狀態 2: 等待進入鬧鐘倒計時 ---------- '''
        elif current_timer_state == "WAIT_TO_ALARM":
            cv2.putText(img_bgr, "Detect:", (390, 70), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 4)
            cv2.putText(img_bgr, "00:00", (430, 130), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 4)
            elapsed_wait = current_time - wait_time
            if elapsed_wait >= Delay_second:
                print("Wait for ALARM Timer", end='\n\n')
                wait_time = None
                alarm_start_time = None
                current_timer_state = "ALARM"

            ''' ---------- 狀態 3: 鬧鐘倒計時 ---------- '''
        elif current_timer_state == "ALARM":
            if not pygame.mixer.music.get_busy():
                try:
                    pygame.mixer.music.play(-1)
                except Exception as e:
                    print("Error playing alarm sound:", e)

            if alarm_start_time is None:
                alarm_start_time = current_time

            elapsed_time = current_time - alarm_start_time
            remaining_time = max(0, alarm - int(elapsed_time))
            
            ''' ---------- 鬧鐘閃爍提示 ---------- '''
            if (current_time % 2.0) < 1.0:
                cv2.putText(img_bgr, "!!! Eat Medicine !!!", (50, 350), cv2.FONT_HERSHEY_SIMPLEX, 4, (0, 0, 255), 10)

            min, secs = divmod(remaining_time, 60)
            alarm_timer = '{:02d}:{:02d}'.format(min, secs)
            cv2.putText(img_bgr, "Alarm:", (400, 70), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 4)
            cv2.putText(img_bgr, alarm_timer, (430, 130), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 4)            

            if remaining_time == 0:
                print("ALARM Timer Finished", end='\n\n')
                pygame.mixer.music.stop()
                alarm_start_time = None
                wait_time = None
                wait_time = current_time
                current_timer_state = "WAIT_TO_DETECT"

            ''' ---------- 狀態 4: 等待進入偵測倒計時 ---------- '''
        elif current_timer_state == "WAIT_TO_DETECT":
            pygame.mixer.music.stop()
            cv2.putText(img_bgr, "Alarm:", (400, 70), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 4)
            cv2.putText(img_bgr, "00:00", (430, 130), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 4)
            elapsed_wait = current_time - wait_time
            if elapsed_wait >= Delay_second:
                print("Wait for Detect Timer", end='\n\n')
                wait_time = None
                detect_start_time = None
                current_timer_state = "DETECT"

        return detect_start_time, alarm_start_time, wait_time, current_timer_state
    
    except Exception as e:
        print("Error in Timer:", e)
        return detect_start_time, alarm_start_time, wait_time, current_timer_state