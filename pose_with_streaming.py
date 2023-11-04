import cv2
from flask import Flask, render_template, Response
from engine.pose_engine import PoseEngine
import engine.utils as utils
import threading
from datetime import datetime


outputFrame = None
lock = threading.Lock()
# initialize a flask object
app = Flask(__name__)

@app.route('/')
def index():
    """Video streaming home page."""
    return render_template('index.html')

def detect_pose():
    # grab global references to the video stream, output frame, and
    # lock variables
    global vs, outputFrame, lock
    
    _MODEL_PATH = "./model/posenet_resnet_50_416_288_16_quant_edgetpu_decoder.tflite"
    engine = PoseEngine(_MODEL_PATH)

        # Frame shape
    _FRAME_WEIGHT, _FRAME_HEIGHT = 1024, 768

    # Threshold of the accuracy
    _THERESHOLD = 0.50
    # Initiating camera instance
    camera = utils.init_camera(_FRAME_WEIGHT, _FRAME_HEIGHT )

    # Initialize frame rate calculation
    frame_rate_calc = 1
    freq = cv2.getTickFrequency()

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc('F','M','P','4')
    video_name= f"./captured_video/{datetime.today().strftime('%Y%m%d%H%M%S')}.avi"

    fps = 6.0
    # Video Recorder instance
    out = cv2.VideoWriter(video_name,fourcc, fps, (_FRAME_WEIGHT, _FRAME_HEIGHT))
    while True:
        # Grab frame from video stream
        image = camera.capture_array()

        # Start timer (for calculating frame rate)
        t1 = cv2.getTickCount()

        _, src_height, src_width, _ = engine.get_input_tensor_shape()
        poses, _ = engine.DetectPosesInImage(image)

        # Draw the lines in the keypoints
        output_image = utils.draw_keypoints_from_keypoints(poses, image, _THERESHOLD, src_width, src_height)
        output_image=cv2.cvtColor(output_image, cv2.COLOR_BGRA2BGR)
        out.write(output_image)

        # Draw framerate in corner of frame
        # Calculate framerate
        t2 = cv2.getTickCount()
        time1 = (t2-t1)/freq
        frame_rate_calc= 1/time1

        # flipping the image for display
        output_image = cv2.flip(output_image, 1)

        # Output show
        cv2.putText(output_image,'FPS: {0:.2f}'.format(frame_rate_calc),(30,50),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,0),2,cv2.LINE_AA)

        with lock:
            outputFrame = output_image.copy()


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
        target=detect_pose, 
    )
    t.daemon = True
    t.start()
    app.run(host='0.0.0.0', port =5000, debug=True, threaded=True, use_reloader=False)
