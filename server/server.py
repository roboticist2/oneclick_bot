import io
from keras.models import load_model
import socket
import struct
import time
from PIL import Image
import numpy as np
import cv2

model = load_model("keras_model.h5", compile=False)

# Start a socket listening for connections on 0.0.0.0:8000
# (0.0.0.0 means all interfaces)
server_socket = socket.socket()
print("socket initializing...")
server_socket.bind(('0.0.0.0', 8000))
server_socket.listen(0)
#client_socket, addr = server_socket.accept()
print("Waiting for connection...")

connection = server_socket.accept()[0].makefile('rb')
print("connection established")

# Read 4 bytes for the length header
#header_data = connection.read(4)

try:
    start=time.time()
    fc = 0
    while True:

        image_len_data = connection.read(4)
        if not image_len_data:
            print('Image loading fail')
            break

        # 이미지 길이 해석
        image_len = struct.unpack('<L', image_len_data)[0]

        # 이미지 데이터 수신
        image_data = b''

        while len(image_data) < image_len:
            to_read = image_len - len(image_data)
            image_data += connection.read(min(4096, to_read))

        # 이미지 데이터를 넘파이 배열로 변환
        nparr = np.frombuffer(image_data, np.uint8)

        # 넘파이 배열을 OpenCV 프레임으로 변환
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        frame = cv2.resize(frame, (224, 224), interpolation=cv2.INTER_AREA)

        # 프레임 출력
        cv2.imshow('Video', frame)

        image = np.asarray(frame, dtype=np.float32).reshape(1, 224, 224, 3)

        image = (image / 127.5) - 1

        time.sleep(0.1)
        # if fc%6==0:
        #     prediction = model.predict(image)
        #     print("prediction: ",prediction,' time:',round(time.time() - start, 1))
        # fc = fc + 1

        dir = 0  # 방향값
        #client_socket.send(struct.pack('dir', value))
        # 0.1초마다 전송
        #time.sleep(0.1)

        # 'q' 키를 누르면 종료
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            #connection.close()
            server_socket.close()
            cv2.destroyAllWindows()

finally:
    connection.close()
    #client_socket.close()
    server_socket.close()
    cv2.destroyAllWindows()