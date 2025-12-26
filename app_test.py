from flask import Flask, render_template, Response
import cv2
import mediapipe as mp
import numpy as np
import time as t
import os

app = Flask(__name__)

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, model_complexity=1)
mp_drawing = mp.solutions.drawing_utils 

def show_landmarks(image, pose):
    output_image = image.copy()
    imageRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    imageRGB.flags.writeable = False 
    results = pose.process(imageRGB)
    imageRGB.flags.writeable = True
    height, width, _ = image.shape
    landmarks = []
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(image=output_image, landmark_list=results.pose_landmarks,
                                  connections=mp_pose.POSE_CONNECTIONS)
        for landmark in results.pose_landmarks.landmark:
            landmarks.append(np.array([int(landmark.x * width), 
                                       int(landmark.y * height)
                                    #    int(landmark.z * width)
                                       ]))
        # landmarks = np.array(landmarks)
    else:
        landmarks = np.array([])
    return output_image, landmarks

def calculate_angle(landmark1, landmark2, landmark3):

    a = landmark1[:2]
    b = landmark2[:2]
    c = landmark3[:2]

    ba = a - b
    bc = c - b

    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.degrees(np.arccos(np.clip(cosine_angle, -1.0, 1.0)))

    return angle

def detectPose(output_image, landmarks):
    try:

        # 兩嘴角中心點:嘴角左[9]、嘴角右[10]
        mouth_center = (landmarks[mp_pose.PoseLandmark.MOUTH_LEFT.value] + landmarks[mp_pose.PoseLandmark.MOUTH_RIGHT.value]) / 2
        cv2.circle(output_image, (int(mouth_center[0]), int(mouth_center[1])), 8, (0, 255, 0), -1)

        # 左手肘角度:左肩[11]、左肘[13]、左腕[15]
        left_elbow_angle = calculate_angle(landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value], 
                                           landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value], 
                                           landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value])

        # 右手肘角度:右肩[12]、右肘[14]、右腕[16]
        right_elbow_angle = calculate_angle(landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value], 
                                           landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value], 
                                           landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value])

        # 左手中心點:左腕[15]、左小指[17]、左食指[19]、左姆指[21]
        left_hand_center = np.array([landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value],
                                     landmarks[mp_pose.PoseLandmark.LEFT_PINKY.value],
                                     landmarks[mp_pose.PoseLandmark.LEFT_INDEX.value],
                                     landmarks[mp_pose.PoseLandmark.LEFT_THUMB.value]]).mean(axis=0)
        cv2.circle(output_image, (int(left_hand_center[0]), int(left_hand_center[1])), 8, (0, 0, 255), -1)

        # 計算左手中心點與嘴中心點距離
        l_2_m_distance = np.linalg.norm(left_hand_center - mouth_center)

        # 右手中心點:右腕[16]、右小指[18]、右食指[20]、右姆指[22]
        right_hand_center =  np.array([landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value],
                                      landmarks[mp_pose.PoseLandmark.RIGHT_PINKY.value],
                                      landmarks[mp_pose.PoseLandmark.RIGHT_INDEX.value],
                                      landmarks[mp_pose.PoseLandmark.RIGHT_THUMB.value]]).mean(axis=0)
        cv2.circle(output_image, (int(right_hand_center[0]), int(right_hand_center[1])), 8, (0, 0, 255), -1)

        # 計算右手中心點與嘴中心點距離
        r_2_m_distance = np.linalg.norm(right_hand_center - mouth_center)

    except Exception as e:
        print(f"Error in detectPose: {e}")

    output_image = cv2.flip(output_image, 1)
    
    cv2.putText(output_image, f'Left Elbow Angle: {int(left_elbow_angle)}', (30, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    cv2.putText(output_image, f'L_H to M Distance: {int(l_2_m_distance)}', (30, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    cv2.putText(output_image, f'Right Elbow Angle: {int(right_elbow_angle)}', (480, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)    
    cv2.putText(output_image, f'R_H to M Distance: {int(r_2_m_distance)}', (480, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)    

    return output_image

camera_video = None
def cap_real_time():
    global camera_video
    try:
        camera_video = cv2.VideoCapture(0)
        camera_video.set(3, 1024)
        camera_video.set(4, 768)
        while camera_video.isOpened():
            ok, frame = camera_video.read()
            if not ok:
                continue

            # frame = cv2.flip(frame, 1)
            frame_height, frame_width, _ = frame.shape
            frame = cv2.resize(frame, (int(frame_width * (650 / frame_height)), 650))

            frame, landmarks = show_landmarks(frame, pose)
            if landmarks:
                frame = detectPose(frame, landmarks)
            else:
                cv2.putText(frame, "No Pose Detected", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            ''' -------- if wanna put other thing, must after this line -------- '''            
            # 目前時間
            current_time = t.localtime()
            format_time = t.strftime("%Y-%m-%d %A %H:%M:%S", current_time)
            now = format_time
            cv2.putText(frame, f"{now}", (80, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 0), 3)

            ret, jpeg = cv2.imencode('.jpg', frame)
            frame = jpeg.tobytes()
            yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
    except Exception as e:
        print(f"Error in cap_real_time: {e}")
    finally:
        if camera_video is not None and camera_video.isOpened():
            camera_video.release()
            cv2.destroyAllWindows()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/stop', methods=['POST'])
def stop():
    global camera_video
    if camera_video is not None and 'camera_video' in globals():
        if camera_video.isOpened():
            camera_video.release()
            # os._exit(0)
    # return render_template('index.html')

@app.route('/cap_in_html')
def cap_in_html():
    return Response(cap_real_time(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=7337)
