import cv2
import helper.utils as utils
import helper.interpreter as interpreter_utils
from datetime import datetime
from flask import Flask, render_template, Response

# The keypoints of model
_NUM_KEYPOINTS = 17

# Model Path
_MODEL_PATH = "./model/posenet_resnet_50_416_288_16_quant_edgetpu_decoder.tflite"

# Frame shape
_FRAME_WEIGHT, _FRAME_HEIGHT = 1024, 768

# Threshold of the accuracy
_THERESHOLD = 0.50

app = Flask(__name__)

@app.route('/')
def index():
    """Video streaming home page."""
    return render_template('index.html')

def gen():
  
  # Initating Interpreter
  interpreter = interpreter_utils.init_interpreter(_MODEL_PATH)

  # Initiating camera instance
  camera = utils.init_camera(_FRAME_WEIGHT, _FRAME_HEIGHT )

  # Initialize frame rate calculation
  frame_rate_calc = 1
  freq = cv2.getTickFrequency()

  # Define the codec and create VideoWriter object
  fourcc = cv2.VideoWriter_fourcc('F','M','P','4')
  video_name= f"./captured_video/{datetime.today().strftime('%Y%m%d%H%M%S')}.avi"
  print(video_name)
  fps = 6.0
  # Video Recorder instance
  out = cv2.VideoWriter(video_name,fourcc, fps, (_FRAME_WEIGHT, _FRAME_HEIGHT))
  
  while True:
    # Grab frame from video stream
    image = camera.capture_array()

    # Start timer (for calculating frame rate)
    t1 = cv2.getTickCount()

    # Transformed Images according to needs
    transformed_image = interpreter_utils.transform_image_for_interpreture(image, interpreter)
    
    # Getting Outputs
    poses, (src_width, src_height) = interpreter_utils.get_interpreter_output(interpreter, transformed_image)

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
    ret, jpeg = cv2.imencode('.jpg', output_image)
    frame=jpeg.tobytes()
    yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
    # cv2.imshow('Pose detector', output_image)

    # # Key to quite display
    # if cv2.waitKey(1) == ord('q'):
    #     break
  
  # Clean up
  out.release()
  cv2.destroyAllWindows()

@app.route('/video_feed')
def video_feed():
    """Video streaming route. Put this in the src attribute of an img tag."""
    return Response(gen(),mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__': 
    app.run(host='0.0.0.0', port =5000, debug=True, threaded=True)