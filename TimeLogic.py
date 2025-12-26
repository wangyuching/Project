import cv2
import time as t
import pygame
import os

detect = 5
alarm = 3
detect_start_time = None
alarm_start_time = None
wait_time = None
current_timer_state = "DETECT"
Delay_second = 1.0

base_dir = os.path.dirname(os.path.abspath(__file__))
alarm_sound_path = os.path.join(base_dir, "static", "alarm3.mp3")

pygame.mixer.init()
try:
    pygame.mixer.music.load(alarm_sound_path)
except Exception as e:
    print(f"載入失敗，請檢查路徑：{alarm_sound_path}")
    print("Error loading alarm sound:", e)

class timer:
    def __init__(self):
        self.detect = 5
        self.alarm = 3
        self.delay_second = 1.0
    
        self.detect_start_time = None
        self.alarm_start_time = None
        self.wait_time = None
        self.current_timer_state = "DETECT"
    
        pygame.mixer.init()
        try:
            pygame.mixer.music.load("./alarm3.mp3")
        except Exception as e:
            print("Error loading alarm sound:", e)

    def update(self, frame, is_eating_medicine):
        try:
            # 取得目前時間戳
            current_time = t.time()

            # 狀態 0: 判斷有無吃藥動作
            if is_eating_medicine:
                pygame.mixer.music.stop()
                self.detect_start_time = None
                self.alarm_start_time = None
                self.wait_time = None
                self.current_timer_state = "DETECT"

            # 狀態 1: 偵測倒數
            elif self.current_timer_state == "DETECT":
                if self.detect_start_time is None:
                    self.detect_start_time = current_time

                remaining_time = max(0, self.detect - int(current_time - self.detect_start_time))
                
                self._draw_status(frame, self.current_timer_state, remaining_time)
                if remaining_time == 0:
                    self.detect_start_time = None
                    self.wait_time = None
                    self.wait_time = current_time
                    self.current_timer_state = "WAIT_TO_ALARM"

            # 狀態 2: 等待進入鬧鐘
            elif self.current_timer_state == "WAIT_TO_ALARM":
                self._draw_status(frame, "DETECT", 0)
                if (current_time - self.wait_time) >= self.delay_second:
                    self.wait_time = None
                    self.alarm_start_time = None
                    self.current_timer_state = "ALARM"

            #狀態 3: 鬧鐘倒計時
            elif self.current_timer_state == "ALARM":
                if not pygame.mixer.music.get_busy():
                    try:
                        pygame.mixer.music.play(-1)
                    except Exception as e:
                        print("Error playing alarm sound:", e)

                if self.alarm_start_time is None:
                    self.alarm_start_time = current_time

                remaining_time = max(0, self.alarm - int(current_time - self.alarm_start_time))
                
                # 鬧鐘閃爍提示
                if (current_time % 2.0) < 1.0:
                    cv2.putText(frame, "!!! Eat Medicine !!!", (50, 350), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 255), 5)

                self._draw_status(frame, "ALARM", remaining_time)

                if remaining_time == 0:
                    pygame.mixer.music.stop()
                    self.alarm_start_time = None
                    self.wait_time = None
                    self.wait_time = current_time
                    self.current_timer_state = "WAIT_TO_DETECT"

            # 狀態 4: 等待重新偵測
            elif self.current_timer_state == "WAIT_TO_DETECT":
                pygame.mixer.music.stop()
                self._draw_status(frame, "ALARM", 0)
                if (current_time - self.wait_time) >= self.delay_second:
                    self.wait_time = None
                    self.detect_start_time = None
                    self.current_timer_state = "DETECT"
    
        except Exception as e:
            print("Error in Timer:", e)
        
        return frame
    
    def _draw_status(self, frame, current_timer_state, remaining_time):
        min, secs = divmod(remaining_time, 60)
        detect_timer = '{:02d}:{:02d}'.format(min, secs)
        cv2.putText(frame, current_timer_state, (390, 70), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 4)
        cv2.putText(frame, detect_timer, (430, 130), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 4)
