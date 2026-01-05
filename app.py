from flask import Flask, render_template, Response
from camera import camera_stream

app = Flask(__name__)

camera_video = None
def get_camera_video():
    global camera_video
    if camera_video is None:
        camera_video = camera_stream()
    return camera_video

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/cap_in_html')
def cap_in_html():
    cam = get_camera_video()
    return Response(cam.cap_real_time(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/stop', methods=['POST'])
def stop():
    global camera_video
    if camera_video:
        camera_video.release()
        camera_video = None
    return "Camera stream stopped."
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=7337)
