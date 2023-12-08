import io
from keras.models import load_model
import socket
import struct
import time
import numpy as np
import cv2
import threading
from pynput import keyboard

model = load_model("keras_model.h5", compile=False)
shared_image = None
keyValue = -1
keyValue2 = "0"
server_socket = socket.socket()
print("Socket initializing...")

#%% keyboard flags
now = 0
go_flag = 0
left_flag = 0
right_flag = 0
back_flag = 0
exit_flag = 0

current_pressed = set()

def getkeyboard():
    with keyboard.Listener(on_press=on_press, on_release=on_release) as listener:
        listener.join()

def on_press(key):

    global go_flag, left_flag, right_flag, back_flag, exit_flag
    current_pressed.add(key)
    # print('Key %s pressed' % current_pressed)
    try:
        if key == keyboard.Key.esc:
            print('Exit is pressed')
            exit_flag=1
        if key.char == 'i':
            # print('go is pressed')
            go_flag=1
        if key.char == 'j':
            # print('left is pressed')
            left_flag=1
        if key.char == 'l':
            # print('right is pressed')
            right_flag=1
        if key.char == 'k':
            # print('right is pressed')
            back_flag=1

    except:
        pass

# 키보드 뗄 때 실행
def on_release(key):
    global go_flag, left_flag, right_flag, back_flag, exit_flag
    # print('Key %s released' %key)
    # if key == keyboard.Key.esc:
    #     return False
    if key in current_pressed:
        current_pressed.remove(key)

    try:
        if key.char == 'i':
            # print('go is released')
            go_flag=0
        if key.char == 'j':
            # print('left is released')
            left_flag=0
        if key.char == 'l':
            # print('right is released')
            right_flag=0
        if key.char == 'k':
            # print('right is pressed')
            back_flag=0
    except:
        pass

def start_server():
    HOST = '0.0.0.0'
    PORT = 8000

    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.bind((HOST, PORT))
    server_socket.listen(1)
    print('Server is listening...')

    while True:
        print('Waiting for a connection...')
        connection, addr = server_socket.accept()
        print('Connection established with', addr)
        video_stream = connection.makefile('rb')
        client_thread = threading.Thread(target=handle_client_connection, args=(video_stream, addr))
        client_thread.start()

        send_thread = threading.Thread(target=send_dirct, args=(connection,))
        send_thread.start()

        getkey_thread = threading.Thread(target=getkeyboard)
        getkey_thread.start()

def handle_client_connection(video_stream, addr):
    global shared_image
    global keyValue
    try:
        print("Connection established ")
        start = time.time()
        while True:
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
                print('User quit')
                server_socket.close()
                cv2.destroyAllWindows()
                break

    except (ConnectionResetError, ConnectionAbortedError):
        print('Connection closed with', addr)
        video_stream.close()
        cv2.destroyAllWindows()
def send_dirct(connection):
    global shared_image
    global go_flag, back_flag,left_flag,right_flag

    while True:
        try:
            if shared_image is None:
                dirct = 3
            else:
                prediction = model.predict(shared_image, verbose=0)
                dirct = np.argmax(prediction[0])
                #dirct = 1

            connection.send(struct.pack('iiii', go_flag, back_flag, left_flag, right_flag))

        except ConnectionResetError:
            while True:
                try:
                    None
                except ConnectionResetError:
                    pass  # 다시 연결되기를 대기

if __name__ == "__main__":
    start_server()