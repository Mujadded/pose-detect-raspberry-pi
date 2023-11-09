# Pose Detection System

This repository contains code for a real-time pose detection system that uses PoseNet, a deep learning model, to detect human poses in images or video streams. The system has two main components:

1. **Web-Based Pose Detection**: A Flask-based web application that streams live video from a camera and displays it in a web browser. It also runs pose detection on the video feed and renders the detected poses on the web page in real-time. The pose detection process is performed using the PoseNet model.

2. **Local Pose Detection**: A command-line script that captures video from a camera and displays it using OpenCV. It also performs pose detection on the video feed, displaying the detected poses in a graphical window.

## Installation

To run this code, you need to set up your Python environment and install the necessary dependencies. You can do this by creating a virtual environment and installing the required packages listed in the `requirements.txt` file.

```bash
pip install -r requirements.txt
```

## Usage

### Web-Based Pose Detection

1. To start the web-based pose detection application, run the `stream_on_web.py` script. This will start a Flask web server that serves the video stream and processes pose detection.

```bash
python stream_on_web.py
```

2. Open a web browser and navigate to `http://localhost:5000` to access the live video stream with pose detection. Press 'q' to quit the application.

### Local Pose Detection

1. To run the local pose detection script, execute the `stream_on_cv.py` script.

```bash
python stream_on_cv.py
```

2. This will open a window displaying the live video feed from your camera with real-time pose detection. Press 'q' to quit the application.

## Model and Configuration

The PoseNet model used for pose detection is loaded from the "posenet_resnet_50_416_288_16_quant_edgetpu_decoder.tflite" file, which should be present in the "PoseNet/model" directory.

You can adjust various parameters like the frame size, accuracy threshold, and video recording settings in the `detect_pose` function in the `posenet/pose.py` script.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgments

The PoseNet model used in this project is based on Google's MediaPipe Pose model. The code has been adapted and modified for this project. For more details on the original model, visit [MediaPipe](https://google.github.io/mediapipe/solutions/pose).

## Author

This project was created by [Your Name]. Feel free to reach out for questions or contributions.

Enjoy exploring and using the real-time pose detection system!