import cv2
import matplotlib
from picamera2 import Picamera2
from PoseNet.engine.pose_engine import KeypointType

# Defining the edges we will draw line to
EDGES = (
    (KeypointType.NOSE, KeypointType.LEFT_EYE),
    (KeypointType.NOSE, KeypointType.RIGHT_EYE),
    (KeypointType.LEFT_EYE, KeypointType.LEFT_EAR),
    (KeypointType.RIGHT_EYE, KeypointType.RIGHT_EAR),
    (KeypointType.NOSE, KeypointType.LEFT_SHOULDER),
    (KeypointType.NOSE, KeypointType.RIGHT_SHOULDER),
    (KeypointType.LEFT_SHOULDER, KeypointType.LEFT_ELBOW),
    (KeypointType.RIGHT_SHOULDER, KeypointType.RIGHT_ELBOW),
    (KeypointType.LEFT_ELBOW, KeypointType.LEFT_WRIST),
    (KeypointType.RIGHT_ELBOW, KeypointType.RIGHT_WRIST),
    (KeypointType.LEFT_SHOULDER, KeypointType.LEFT_HIP),
    (KeypointType.RIGHT_SHOULDER, KeypointType.RIGHT_HIP),
    (KeypointType.LEFT_HIP, KeypointType.LEFT_KNEE),
    (KeypointType.RIGHT_HIP, KeypointType.RIGHT_KNEE),
    (KeypointType.LEFT_KNEE, KeypointType.LEFT_ANKLE),
    (KeypointType.RIGHT_KNEE, KeypointType.RIGHT_ANKLE)
)

# EDGES = (
#     (KeypointType.NOSE, KeypointType.LEFT_EYE), 
#     (KeypointType.NOSE, KeypointType.RIGHT_EYE),
#     (KeypointType.NOSE, KeypointType.LEFT_EAR),
#     (KeypointType.NOSE, KeypointType.RIGHT_EAR),
#     (KeypointType.LEFT_EAR, KeypointType.LEFT_EYE),
#     (KeypointType.RIGHT_EAR, KeypointType.RIGHT_EYE),
#     (KeypointType.LEFT_EYE, KeypointType.RIGHT_EYE),
#     (KeypointType.LEFT_SHOULDER, KeypointType.RIGHT_SHOULDER),
#     (KeypointType.LEFT_SHOULDER, KeypointType.LEFT_ELBOW),
#     (KeypointType.LEFT_SHOULDER, KeypointType.LEFT_HIP),
#     (KeypointType.RIGHT_SHOULDER, KeypointType.RIGHT_ELBOW),
#     (KeypointType.RIGHT_SHOULDER, KeypointType.RIGHT_HIP),
#     (KeypointType.LEFT_ELBOW, KeypointType.LEFT_WRIST),
#     (KeypointType.RIGHT_ELBOW, KeypointType.RIGHT_WRIST),
#     (KeypointType.LEFT_HIP, KeypointType.RIGHT_HIP),
#     (KeypointType.LEFT_HIP, KeypointType.LEFT_KNEE),
#     (KeypointType.RIGHT_HIP, KeypointType.RIGHT_KNEE),
#     (KeypointType.LEFT_KNEE, KeypointType.LEFT_ANKLE),
#     (KeypointType.RIGHT_KNEE, KeypointType.RIGHT_ANKLE),
# )

def init_camera(width, height):
  # Get PiCamera2 library
  picam2 = Picamera2()
  picam2.configure(picam2.create_preview_configuration(main={"format": 'XRGB8888', "size": (width, height)}))
  picam2.start()

  return picam2

def draw_keypoints_from_keypoints(poses, image, threshold, src_width,src_height):
    # Getting the height and width of the orginal image
    height, width = image.shape[0], image.shape[1] 
    
    # Calculating the scale of which the points will be scaled to
    scale_x, scale_y = width/src_width , height/src_height
    
    # Looping through the poses
    for pose in poses:
        # Saving the point names with scores so that a one pose will not be
        # Connected to the next
        xys = {}
        for label, keypoint in pose.keypoints.items():
            # Checking threshold set
            if keypoint.score < threshold: continue

            kp_x = int((keypoint.point[0]) * scale_x)
            kp_y = int((keypoint.point[1]) * scale_y)

            cv2.circle(
                image,
                (kp_x,kp_y), 
                3,
                (0, 0, 255),
                thickness=-1,
                lineType=cv2.FILLED
            )

            xys[label] = (kp_x, kp_y)

        for ie, (a, b) in enumerate(EDGES):
            
            # Random color generations for the lines
            rgb = matplotlib.colors.hsv_to_rgb([
                        ie/float(len(EDGES)), 1.0, 1.0
                    ])
            rgb = rgb*255
            
            # so that we dont draw lines for open end (where the other dot is missing)
            if a not in xys or b not in xys: continue

            ax, ay = xys[a]
            bx, by = xys[b]
            cv2.line(
                image,
                (ax, ay),
                (bx, by),
                tuple(rgb), 
                2, 
                lineType=cv2.LINE_AA
            )

    return image
