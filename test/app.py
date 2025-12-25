from flask import Flask, render_template, Response
import cv2
app = Flask(__name__)
cap = None

def cap_real_time():
    global cap
    cap = cv2.VideoCapture(0)
    while True:
        ret, img = cap.read()
        if not ret:
            break
        else:
            ret, buffer = cv2.imencode('.jpg', img)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/start', methods=['POST'])
def start():
    return render_template('index.html')

@app.route('/stop',methods=['POST'])
def stop():
    global cap
    if cap is not None and 'cap' in globals():
        if cap.isOpened():
            cap.release()
    return render_template('stop.html')

@app.route('/cap_in_html')
def cap_in_html():
    return Response(cap_real_time(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__=='__main__':
    app.run(debug=True)