import cv2
from flask import Flask, render_template, Response
from io import BytesIO
from math import log
import numpy as np
import time
import sys


MAX_FILE_SIZE = 62914560 # bytes
MAX_FILE_SIZE = int(log(MAX_FILE_SIZE, 2)+1)

app = Flask(__name__)

@app.route('/')
def index():
    """Video streaming home page."""
    return render_template('index.html')

def read(buffer, number_of_bytes):
    output = b""
    while len(output) < number_of_bytes:
        output += buffer.read(number_of_bytes - len(output))
    assert len(output) == number_of_bytes, "An error occured."
    return output


def read_file(buffer):
    # Read `MAX_FILE_SIZE` number of bytes and convert it to an int
    # So that we know the size of the file comming in
    length = int(read(buffer, MAX_FILE_SIZE))

    # Here you can switch to a different file every time `writer.py`
    # Sends a new file
    data = read(buffer, length)

    # Read a byte so that we know if it is the last file
    file_ended = read(buffer, 1)

    return data, (file_ended == b"1")

def gen():
    while True:
      data, last_file = read_file(sys.stdin.buffer)
      jpeg = cv2.imdecode(np.frombuffer(BytesIO(data).read(), np.uint8),
                            cv2.IMREAD_UNCHANGED)
      frame=jpeg.tobytes()
      yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
      if last_file:
          break;

@app.route('/video_feed')
def video_feed():
    """Video streaming route. Put this in the src attribute of an img tag."""
    return Response(gen(),mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__': 
    app.run(host='0.0.0.0', port =5000, debug=True, threaded=True)
