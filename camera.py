import cv2
import mediapipe as mp
import time as t
import DrawUtil
import TimeLogic

class camera_stream:

    def __init__(self, video_source=0):
        self.camera = cv2.VideoCapture(video_source)
        self.camera.set(3, 1024)
        self.camera.set(4, 768)
        self.timer = TimeLogic.timer()

    def __del__(self):
        self.release()

    def release(self):
        if self.camera is not None and self.camera.isOpened():
            self.camera.release()
            cv2.destroyAllWindows()

    def cap_real_time(self):
        try:

            while self.camera.isOpened():
                ok, frame = self.camera.read()
                if not ok:
                    continue
                
                frame_height, frame_width, _ = frame.shape
                frame = cv2.resize(frame, (int(frame_width * (650 / frame_height)), 650))

                frame, landmarks = DrawUtil.show_landmarks(frame, DrawUtil.pose)

                # 偵測吃藥動作
                is_eating_medicine = False
                if landmarks:
                    frame, is_eating_medicine = DrawUtil.detectPose(frame, landmarks)
                else:
                    cv2.putText(frame, "No Pose Detected", (80, 400), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 255), 7)
                
                ''' -------- if wanna put other thing, must after this line -------- '''            
                # 目前時間
                current_time = t.localtime()
                format_time = t.strftime("%Y-%m-%d %A %H:%M:%S", current_time)
                now = format_time
                cv2.putText(frame, f"{now}", (80, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 0), 3)

                # ''' ---------- 倒數計時器 ---------- '''
                frame = self.timer.update(frame, is_eating_medicine)   

                ret, jpeg = cv2.imencode('.jpg', frame)
                frame = jpeg.tobytes()
                yield (b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        except Exception as e:
            print(f"Error in cap_real_time: {e}")
        finally:
            self.release()