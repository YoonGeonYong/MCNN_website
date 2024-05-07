import time
import cv2
import requests

cap = cv2.VideoCapture(0)

while True:
    time.sleep(2)
    ret, frame = cap.read()  # frame 읽기
    if ret:
        # encoding
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        _, encoded = cv2.imencode('.jpg', gray, params=[cv2.IMWRITE_JPEG_QUALITY, 10]) # jpeg 10%
        byted = encoded.tobytes()

        # send to server
        response = requests.post('http://localhost:5000/recv_send', data=byted)
        '''
            tobytes()       : array -> bytes
            frombuffer()    : bytes -> array
        '''

        # quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()