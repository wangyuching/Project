import cv2
import mediapipe as mp
import time as t
import DrawUtil
from datetime import datetime as dt

start = dt.now()

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

w, h = 1000, 750
detect, alert = 30, 5
cap = cv2.VideoCapture(0)

with mp_pose.Pose(min_detection_confidence=0.5,min_tracking_confidence=0.5) as pose:

    while cap.isOpened():
        ret, frame = cap.read()
        img_bgr = cv2.resize(frame, (w, h))
        now = dt.now().strftime('%Y-%m-%d %A %H:%M:%S')

        if not ret:
            print("Ignoring empty camera frame.")
            break

        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        results = pose.process(img_rgb)

        (DrawUtil_img_bgr, 
         left_elbow_angle, right_elbow_angle, 
         left_distence, right_distence) = DrawUtil.frame(img_bgr, results, w, h)

        try:
            ''' ---------- 計時 ---------- '''


            ''' ---------- 判斷吃藥動作 ---------- '''
            if left_elbow_angle < 50 and left_distence < 150 :
                cv2.putText(DrawUtil_img_bgr, "Eat Meduicine", (50, 350), cv2.FONT_HERSHEY_SIMPLEX, 5, (255, 0, 0), 15)
            elif right_elbow_angle < 50 and right_distence < 150 :
                cv2.putText(DrawUtil_img_bgr, "Eat Meduicine", (50, 350), cv2.FONT_HERSHEY_SIMPLEX, 5, (255, 0, 0), 15)
            else:
                pass

            cv2.putText(DrawUtil_img_bgr, f"{now}", (50, 350), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 0), 5)

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

end = dt.now()
print(f"start time: {start}")
print(f"end time: {end}")
print(f"Execution time: {end - start} seconds")