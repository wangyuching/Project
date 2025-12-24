from flask import Flask, render_template, Response
import cv2
app = Flask(__name__)

def cap_real_time():
    global cap
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        cv2.imshow('Real-Time Video', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/start')
def start():
    return render_template('start.html')

@app.route('/stop',methods=['POST'])
def stop():
    if cap.isOpened():
        cap.release()
        cv2.destroyAllWindows()
    return render_template('stop.html')

@app.route('/video_capture')
def video_capture():
    return Response(cap_real_time(), mimetype='multipart/x-mixed-replace; boundary=frame')
if __name__=='__main__':
    app.run(debug=True,use_reloader=False, port=8000)