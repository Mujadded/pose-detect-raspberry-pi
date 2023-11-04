import cv2
import engine.utils as utils
from datetime import datetime
from engine.pose_engine import PoseEngine

# Model Path
_MODEL_PATH = "./model/posenet_resnet_50_416_288_16_quant_edgetpu_decoder.tflite"

# Frame shape
_FRAME_WEIGHT, _FRAME_HEIGHT = 1024, 768

# Threshold of the accuracy
_THERESHOLD = 0.50



def detect_pose(callback_function, quit_on_key=True):
  # Initating Interpreter
  engine = PoseEngine(_MODEL_PATH)

  # Initiating camera instance
  camera = utils.init_camera(_FRAME_WEIGHT, _FRAME_HEIGHT )

  # Initialize frame rate calculation
  frame_rate_calc = 1
  freq = cv2.getTickFrequency()

  # Define the codec and create VideoWriter object
  fourcc = cv2.VideoWriter_fourcc('F','M','P','4')
  video_name= f"./PoseNet/captured_video/{datetime.today().strftime('%Y%m%d%H%M%S')}.avi"

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
    # cv2.imshow('Pose detector', output_image)
    # cv2.imwrite('./buffer_image/test.jpg', output_image)
    callback_function(output_image)

    # Key to quite display
    if cv2.waitKey(1) == ord('q') and quit_on_key:
        break
  
  # Clean up
  out.release()
  cv2.destroyAllWindows()