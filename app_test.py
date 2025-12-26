from flask import Flask, render_template, Response
import cv2
import mediapipe as mp
import numpy as np
import time as t
import math

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
                                       ]))
        landmarks = np.array(landmarks)
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
    if landmarks.size == 0:
        cv2.putText(output_image, "No Pose Detected", (10, 30), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)
        return output_image

    try:
        left_elbow_angle = 0
        right_elbow_angle = 0
        left_distence = 0
        right_distence = 0

        # 嘴角左[9]、嘴角右[10]
        mouth_center = (landmarks[9] + landmarks[10]) / 2
        cv2.circle(output_image, (int(mouth_center[0]), int(mouth_center[1])), 8, (0, 255, 0), -1)
    
    except Exception as e:
        print(f"Error in detectPose: {e}")

    return output_image

# def classifyPose(landmarks, output_image):
#     label = 'Unknown Pose'
#     color = (0, 0, 255)
#     left_elbow_angle = calculate_angle(landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value],
#                                       landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value],
#                                       landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value])
#     right_elbow_angle = calculate_angle(landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value],
#                                        landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value],
#                                        landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value])   
#     left_shoulder_angle = calculate_angle(landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value],
#                                          landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value],
#                                          landmarks[mp_pose.PoseLandmark.LEFT_HIP.value])
#     right_shoulder_angle = calculate_angle(landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value],
#                                           landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value],
#                                           landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value])
#     left_knee_angle = calculate_angle(landmarks[mp_pose.PoseLandmark.LEFT_HIP.value],
#                                      landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value],
#                                      landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value])
#     right_knee_angle = calculate_angle(landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value],
#                                       landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value],
#                                       landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value])
#     if (165 < left_knee_angle < 195) and (165 < right_knee_angle < 195) \
#         and (130 < left_elbow_angle < 180) and (175 < right_elbow_angle < 220) \
#         and (100 < left_shoulder_angle < 200) and (50 < right_shoulder_angle < 130):
#         label = 'T Pose'
#     if left_elbow_angle > 165 and left_elbow_angle < 195 and right_elbow_angle > 165 and right_elbow_angle < 195:
#         if left_shoulder_angle > 80 and left_shoulder_angle < 110 and right_shoulder_angle > 80 and right_shoulder_angle < 110:
#             if left_knee_angle > 165 and left_knee_angle < 195 or right_knee_angle > 165 and right_knee_angle < 195:
#                 if left_knee_angle > 90 and left_knee_angle < 120 or right_knee_angle > 90 and right_knee_angle < 120:
#                     label = 'Warrior II Pose' 
#     if left_knee_angle > 165 and left_knee_angle < 195 or right_knee_angle > 165 and right_knee_angle < 195:
#         if left_knee_angle > 315 and left_knee_angle < 335 or right_knee_angle > 25 and right_knee_angle < 45:
#             label = 'Tree Pose'
#     if label != 'Unknown Pose':
#         color = (0, 255, 0)  
#     cv2.putText(output_image, label, (10, 30), cv2.FONT_HERSHEY_PLAIN, 2, color, 2)
    
#     return output_image, label

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
            frame = cv2.flip(frame, 1)
            frame_height, frame_width, _ = frame.shape
            frame = cv2.resize(frame, (int(frame_width * (650 / frame_height)), 650))
            frame, landmarks = show_landmarks(frame, pose)
            # if landmarks.size > 0:
            #     frame = detectPose(frame, landmarks)
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
    # return render_template('index.html')

@app.route('/cap_in_html')
def cap_in_html():
    return Response(cap_real_time(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=7337)
