import tflite_runtime.interpreter as tflite
from pycoral.adapters import common
from pycoral.utils import edgetpu
import cv2
import numpy as np
from tflite_runtime.interpreter import load_delegate

# POSENET Things
EDGETPU_SHARED_LIB = 'libedgetpu.so.1'
POSENET_SHARED_LIB = 'model/posenet_decoder.so'

def init_interpreter(path: str):
  # Initiating Interpreter 
  # https://coral.ai/docs/reference/py/pycoral.utils/#pycoral.utils.edgetpu.make_interpreter
  # https://coral.ai/docs/edgetpu/tflite-python/#inferencing-example
  edgetpu_delegate = load_delegate(EDGETPU_SHARED_LIB)
  posenet_decoder_delegate = load_delegate(POSENET_SHARED_LIB)
  interpreter = tflite.Interpreter(
    path,
    experimental_delegates=[edgetpu_delegate,posenet_decoder_delegate]
  )
  interpreter.allocate_tensors()

  return interpreter

def get_output_tensor(interpreter, idx):
    """Returns output tensor view."""
    return np.squeeze(interpreter.tensor(
        interpreter.get_output_details()[idx]['index'])())

def parse_output(interpreter):
  """Parses interpreter output tensors and returns decoded poses."""
  keypoints = get_output_tensor(interpreter, 0)
  # keypoint_scores = get_output_tensor(interpreter, 1)
  pose_scores = get_output_tensor(interpreter, 2)
  num_poses = get_output_tensor(interpreter, 3)
  # poses = []
  for i in range(int(num_poses)):
      pose_score = pose_scores[i]
      pose_keypoints = []
      for j, point in enumerate(keypoints[i]):
          y, x = point
          pose_keypoints.append([x,y,pose_score])
  return pose_keypoints

def get_interpreter_output(interpreter, input_image):
    # Setting up interpreter inputs
    # common.set_input(interpreter, input_image)
    # interpreter.invoke()

    # #Getting output
    # output = common.output_tensor(interpreter, 0)
    input_data = np.asarray(input_image)
    edgetpu.run_inference(interpreter, input_data.flatten())
    return parse_output(interpreter)

def transform_image_for_interpreture(image, interpreter):
    # getting required input size
    width, height = common.input_size(interpreter)

    #Converting Image to RGB
    frame_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    #Resizing Images
    frame_resized = cv2.resize(frame_rgb, (width, height))

    #inserting a new dim basically unsqueeze of pytorch
    input_data = np.expand_dims(frame_resized, axis=0)

    return input_data