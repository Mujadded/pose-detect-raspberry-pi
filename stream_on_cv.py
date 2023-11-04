import cv2
from PoseNet.pose import detect_pose

def main():
  detect_pose(display_in_cv_image, quit_on_key=True)

def display_in_cv_image(image):
  cv2.imshow('Pose detector', image)
  

if __name__ == '__main__':
  main()
