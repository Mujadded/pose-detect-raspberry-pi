import cv2
import matplotlib
from picamera2 import Picamera2

edges = [
    (0, 1), (0, 2), (1, 3), (2, 4), (0, 5), (0, 6), (5, 7), (6, 8),
    (7, 9), (8, 10), (5, 11), (6, 12), (11, 13), (12, 14), (13, 15), (14, 16)
]

def init_camera(height,width):
  # Get PiCamera2 library
  picam2 = Picamera2()
  picam2.configure(picam2.create_preview_configuration(main={"format": 'XRGB8888', "size": (height, width)}))
  picam2.start()

  return picam2

def draw_keypoints_from_keypoints(keypoints, image, threshold):
    # the `outputs` is list which in-turn contains the dictionaries
    height, width = image.shape[0], image.shape[1] 
    for edge in edges:
        # proceed to draw the lines if the confidence score is above 0.9
        # if outputs[0]['scores'][i] > 0.9:
        if (keypoints[edge[0]][2] > threshold and keypoints[edge[1]][2] > threshold):
            for p in range(keypoints.shape[0]):
                # draw the keypoints
                cv2.circle(image, (int(keypoints[p, 1]* width-2),int(keypoints[p, 0]* height-2)), 
                            3, (0, 0, 255), thickness=-1, lineType=cv2.FILLED)

                # uncomment the following lines if you want to put keypoint number
                # cv2.putText(image, f"{p}", (int(keypoints[p, 0]+10), int(keypoints[p, 1]-5)),
                #             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

            for ie, e in enumerate(edges):
                # get different colors for the edges
                rgb = matplotlib.colors.hsv_to_rgb([
                    ie/float(len(edges)), 1.0, 1.0
                ])
                rgb = rgb*255
                # join the keypoint pairs to draw the skeletal structure
                cv2.line(image, (int(keypoints[e, 1][0]*width-2),int(keypoints[e, 0][0]*height-2)),
                        (int(keypoints[e, 1][1]*width-2),int(keypoints[e, 0][1]*height-2)),
                        tuple(rgb), 2, lineType=cv2.LINE_AA)
                
        else:
            continue

    return image
