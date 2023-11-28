import io
from keras.models import load_model
import socket
import struct
import time
import numpy as np
import cv2
import threading

model = load_model("keras_model.h5", compile=False)
shared_image = None
keyValue = -1

# Start a socket listening for connections on 0.0.0.0:8000
# (0.0.0.0 means all interfaces)
server_socket = socket.socket()
print("Socket initializing...")

server_socket.bind(('0.0.0.0', 8000))
server_socket.listen(0)
print("Waiting for connection...")

server_socket, addr = server_socket.accept()
video_stream = server_socket.makefile('rb')
print("Connection established")

def send_dirct(server_socket):
    global shared_image
    global keyValue
    carState = -1

    while True:
        if keyValue == 50:
            carState = 1
        if keyValue == 49:
            carState = 0
        if shared_image is None:
            dirct=3
        else:
            prediction = model.predict(shared_image, verbose=0)
            dirct = np.argmax(prediction[0])
            print(prediction, dirct)
        server_socket.send(struct.pack('ii', carState , dirct))

def receive_video(video_stream):
    global shared_image
    global keyValue
    try:
        start=time.time()
        while True:

            #print('time:', round(time.time() - start, 1))

            image_len_data = video_stream.read(4)
            if not image_len_data:
                print('Image loading fail')
                break

            # 이미지 길이 해석
            image_len = struct.unpack('<L', image_len_data)[0]

            # 이미지 데이터 수신
            image_data = b''

            while len(image_data) < image_len:
                to_read = image_len - len(image_data)
                image_data += video_stream.read(min(2048, to_read))

            # 이미지 데이터를 넘파이 배열로 변환
            nparr = np.frombuffer(image_data, np.uint8)

            # 넘파이 배열을 OpenCV 프레임으로 변환
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            frame = cv2.rotate(frame, cv2.ROTATE_180)

            frame = cv2.resize(frame, (224, 224), interpolation=cv2.INTER_AREA)

            # 프레임 출력
            cv2.imshow('Video', frame)
            image = np.asarray(frame, dtype=np.float32).reshape(1, 224, 224, 3)
            image = (image / 127.5) - 1
            shared_image = image

            # 'q' 키를 누르면 종료
            keyValue = cv2.waitKey(1)
            if keyValue & 0xFF == ord('q'):
                server_socket.close()
                cv2.destroyAllWindows()
                break

    finally:
        server_socket.close()
        cv2.destroyAllWindows()

dirct_thread = threading.Thread(target=send_dirct, args=(server_socket,))
dirct_thread.start()

video_thread = threading.Thread(target=receive_video, args=(video_stream,))
video_thread.start()
print('video streaming start')
video_thread.join()
dirct_thread.join()

server_socket.close()
cv2.destroyAllWindows()