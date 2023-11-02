# Lint as: python3
# Copyright 2021 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
r"""Example using PyCoral to estimate a single human pose with Edge TPU MoveNet.

To run this code, you must attach an Edge TPU to the host and
install the Edge TPU runtime (`libedgetpu.so`) and `tflite_runtime`. For
device setup instructions, see coral.ai/docs/setup.

For more details about MoveNet and its best practices, please see
https://www.tensorflow.org/hub/tutorials/movenet

Example usage:
```
bash examples/install_requirements.sh movenet_pose_estimation.py

python3 examples/movenet_pose_estimation.py \
  --model test_data/movenet_single_pose_lightning_ptq_edgetpu.tflite  \
  --input test_data/squat.bmp
```
"""


from PIL import Image
from PIL import ImageDraw
from pycoral.adapters import common
from pycoral.utils.edgetpu import make_interpreter
from picamera2 import Picamera2
import cv2
import numpy as np
import utils

_NUM_KEYPOINTS = 17


def main():
  model_path = "./model/movenet_single_pose_lightning_ptq_edgetpu.tflite"

  interpreter = make_interpreter(model_path)
  interpreter.allocate_tensors()
  width, height = common.input_size(interpreter)
  picam2 = Picamera2()
  picam2.configure(picam2.create_preview_configuration(main={"format": 'XRGB8888', "size": (640, 480)}))
  picam2.start()
  out = cv2.VideoWriter('outpy.avi',-1, 20.0, (height, width))

  while True:
    # Grab frame from video stream
    im = picam2.capture_array()

    frame_rgb = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    frame_resized = cv2.resize(frame_rgb, (width, height))
    input_data = np.expand_dims(frame_resized, axis=0)
    common.set_input(interpreter, input_data)

    interpreter.invoke()

    pose = common.output_tensor(interpreter, 0).copy().reshape(_NUM_KEYPOINTS, 3)
    # print(pose)
    # draw = ImageDraw.Draw(img)
    # width, height = img.size
    # for i in range(0, _NUM_KEYPOINTS):
    #   draw.ellipse(
    #       xy=[
    #           pose[i][1] * width - 2, pose[i][0] * height - 2,
    #           pose[i][1] * width + 2, pose[i][0] * height + 2
    #       ],
    #       fill=(255, 0, 0))
    # img.save(args.output)
    # print('Done. Results saved at', args.output)
    output_image = utils.draw_keypoints_from_keypoints(pose, frame_rgb)
    output_image = cv2.flip(output_image, 1) 
    out.write(output_image)
    cv2.imshow('Object detector', output_image)
        # Press 'q' to quit
    if cv2.waitKey(1) == ord('q'):
        break
  # Clean up
  out.release()
  cv2.destroyAllWindows()

if __name__ == '__main__':
  main()
