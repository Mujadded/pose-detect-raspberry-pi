import cv2
from flask import Flask, render_template, Response
import threading
from pose import detect_pose


outputFrame = None
lock = threading.Lock()
# initialize a flask object
app = Flask(__name__)

@app.route('/')
def index():
    """Video streaming home page."""
    return render_template('index.html')

def render_image(image):
    # grab global references to the video stream, output frame, and
    # lock variables
    global outputFrame, lock
    with lock:
        outputFrame = image.copy()

def call():
    detect_pose(render_image, quit_on_key=False)
      


def gen():
    global outputFrame, lock
    # loop over frames from the output stream
    while True:
        # wait until the lock is acquired
        with lock:
            # check if the output frame is available, otherwise skip
            # the iteration of the loop
            if outputFrame is None:
                continue
            (flag, encodedImage) = cv2.imencode(".jpg", outputFrame)

            # ensure the frame was successfully encoded
            if not flag:
                continue
        yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + bytearray(encodedImage) + b'\r\n')

@app.route('/video_feed')
def video_feed():
    """Video streaming route. Put this in the src attribute of an img tag."""
    return Response(gen(),mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__': 
    # start a thread that will perform motion detection
    t = threading.Thread(
        target=call, 
    )
    t.daemon = True
    t.start()
    app.run(host='0.0.0.0', port =5000, debug=True, threaded=True, use_reloader=False)
