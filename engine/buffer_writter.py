
from math import log
import time
import sys
import cv2


MAX_FILE_SIZE = 62914560 # bytes
MAX_FILE_SIZE = int(log(MAX_FILE_SIZE, 2)+1)


def write_file(buffer, data, last_file=False):
   # Tell `reader.py` that it needs to read x number of bytes.
   length = len(data)
   # We also need to tell `read.py` how many bytes it needs to read.
   # This means that we have reached the same problem as before.
   # To fix that issue we are always going to send the number of bytes but
   # We are going to pad it with `0`s at the start.
   # https://stackoverflow.com/a/339013/11106801
   length = str(length).zfill(MAX_FILE_SIZE)
   with open("./buffer_image/output.txt", "w") as file:
      file.write(length)
   buffer.write(length.encode())

   data = cv2.imencode(".jpg", data)[1]
   # Write the actual data
   buffer.write(data)

   # We also need to tell `read.py` that it was the last file that we send
   # Sending `1` means that the file has ended
   buffer.write(str(int(last_file)).encode())
   buffer.flush()
