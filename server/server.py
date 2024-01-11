import io
from keras.models import load_model
import socket
import struct
import time
import numpy as np
import cv2
import threading
import os
from pynput import keyboard
import datetime

model = load_model("keras_model.h5", compile=False)
shared_image = None
image_for_save = None
server_socket = socket.socket()
print("Socket initializing...")

directory = 'image'
if not os.path.exists(directory):
    os.makedirs(directory)

#%% keyboard flags
now = 0
go_flag = 0
left_flag = 0
right_flag = 0
back_flag = 0
save_flag = 0

current_pressed = set()

def getkeyboard():
    with keyboard.Listener(on_press=on_press, on_release=on_release) as listener:
        listener.join()

def on_press(key):
    global go_flag, left_flag, right_flag, back_flag, save_flag
    current_pressed.add(key)
    try:
        if key == keyboard.Key.esc:
            print('Exit is pressed')
            exit_flag=1
        if key.char == 'i':
            go_flag=1
        if key.char == 'j':
            left_flag=1
        if key.char == 'l':
            right_flag=1
        if key.char == 'k':
            back_flag=1
        if key.char == 's':
            save_flag=1

    except:
        pass

# 키보드 뗄 때 실행
def on_release(key):
    global go_flag, left_flag, right_flag, back_flag, save_flag
    if key in current_pressed:
        current_pressed.remove(key)
    try:
        if key.char == 'i':
            go_flag=0
        if key.char == 'j':
            left_flag=0
        if key.char == 'l':
            right_flag=0
        if key.char == 'k':
            back_flag=0
        if key.char == 's':
            save_flag=0

    except:
        pass

def start_server():
    global shared_image

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
    global image_for_save
    global keyValue
    save_cnt = 0
    save_time = 0
    save_time_flag = 0
    save_time_coef = 2.0
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
            image_for_save = frame

            Cur_time = round(time.time() - start, 2)
            now = datetime.datetime.now()

            time_string = now.strftime("%H%M%S_%f")[:-3]  # 시, 분, 초, 밀리초

            if (Cur_time - save_time > 0.1*save_time_coef) :
                save_time_flag = 1

            if save_time_flag == 1 and save_flag == 1:
                #file_name = f"{directory}/{time_string}_{go_flag}{left_flag}{right_flag}{back_flag}.jpg"
                file_name = f"{directory}/{go_flag}{left_flag}{right_flag}{back_flag}_{time_string}.jpg"
                cv2.imwrite(file_name, frame)
                save_cnt += 1
                print(f"Image saved as {file_name}, time : {Cur_time}, cnt : {save_cnt}")
                save_time_flag = 0 # 저장 플래그 초기화
                save_time = Cur_time #기준 시간 리셋
            else :
                pass

            # 프레임 출력
            cv2.imshow('Video', frame)
            image = cv2.resize(frame, (64, 64))

            # half_height = image.shape[0] // 2  # 이미지 아래 반쪽의 높이
            # bottom_half = image[half_height:, :, :]
            # image = cv2.resize(bottom_half, (64, 64))

            split_ratio = 0.4 #위쪽 기준
            height = image.shape[0]  # 이미지의 높이
            split_height = int(height * split_ratio)  # 원하는 비율에 따라 분할할 높이 계산

            #upper_part = image[:split_height, :, :]  # 위쪽 부분 추출
            lower_part = image[split_height:, :, :]  # 아래쪽 부분 추출
            image = cv2.resize(lower_part, (64, 64))

            image = np.asarray(image, dtype=np.float32).reshape(1, 64, 64, 3) #(1,224,224,3)
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
    global go_flag,left_flag,right_flag,back_flag
    prediction = 3
    # 0 : auto driving, 1: manual driving
    manual = 0

    while True:
        try:
            if shared_image is None:
                None
            else:
                prediction = model.predict(shared_image, verbose=0)
                print(prediction)
                prediction = np.argmax((prediction),axis=1)

            if manual == 0 :
                if prediction == 0 :
                    go_flag = 1
                    left_flag = 0
                    right_flag = 0
                    back_flag = 0
                elif prediction == 1 :
                    go_flag = 1
                    left_flag = 1
                    right_flag = 0
                    back_flag = 0
                elif prediction == 2 :
                    go_flag = 1
                    left_flag = 0
                    right_flag = 1
                    back_flag = 0
            else :
                None

            connection.send(struct.pack('iiii', go_flag, left_flag, right_flag, back_flag))

        except ConnectionResetError:
            while True:
                try:
                    None
                except ConnectionResetError:
                    pass  # 연결 대기

if __name__ == "__main__":
    start_server()