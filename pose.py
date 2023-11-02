from pycoral.adapters import common
import cv2
import helper.utils as utils
import helper.interpreter as interpreter_utils

# The keypoints of model
_NUM_KEYPOINTS = 17

# Model Path
_MODEL_PATH = "./model/movenet_single_pose_lightning_ptq_edgetpu.tflite"

# Frame shape
_FRAME_HEIGHT, _FRAME_WEIGHT = 640, 480

# Threshold of the accuracy
_THERESHOLD = 0.50

def main():
  # Initating Interpreter
  interpreter = interpreter_utils.init_interpreter(_MODEL_PATH)

  # Initiating camera instance
  camera = utils.init_camera(_FRAME_HEIGHT, _FRAME_WEIGHT)

  # Video Recorder instance
  out = cv2.VideoWriter('outpy.avi',-1, 20.0, (_FRAME_HEIGHT, _FRAME_WEIGHT))

  while True:
    # Grab frame from video stream
    image = camera.capture_array()

    # Transformed Images according to needs
    transformed_image = interpreter_utils.transform_image_for_interpreture(image, interpreter)
    
    # Getting Outputs
    keypoints = interpreter_utils.get_interpreter_output(interpreter, transformed_image)

    # Reshapping keypoints
    keypoints = keypoints.reshape(_NUM_KEYPOINTS, 3)

    # Draw the lines in the keypoints
    output_image = utils.draw_keypoints_from_keypoints(keypoints, image, _THERESHOLD)
    out.write(output_image)

    # flipping the image for display
    output_image = cv2.flip(output_image, 1)

    # Output show
    cv2.imshow('Pose detector', output_image)

    # Key to quite display
    if cv2.waitKey(1) == ord('q'):
        break
  
  # Clean up
  out.release()
  cv2.destroyAllWindows()

if __name__ == '__main__':
  main()
