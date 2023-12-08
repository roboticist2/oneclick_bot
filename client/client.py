import io
import socket
import struct
import time
import cv2
import threading

import final

# Connect a client socket to my_server:8000 1
# Change 'my_server' to the hostname(ip) of your server
client_socket = socket.socket()
client_socket.connect(('172.30.1.66', 8000))

connection = client_socket.makefile('wb')

def receive_dirct(client_socket):
        start=time.time()
        time0=0
        fps_temp=0
        key_temp=0
        FPS=0
        
        go_flag=0
        back_flag=0
        left_flag=0
        right_flag=0
        
        while True:
                data = client_socket.recv(16)
                go_flag, back_flag, left_flag, right_flag = struct.unpack('<IIII', data)
                if(round(time0,0)==round(time.time())):
                        fps_temp+=1
                else:
                        FPS=fps_temp
                        fps_temp=0
                print("Time : ",round(time.time()-start,1),", FPS : ", FPS, "Go : ", go_flag, ", Back : ",back_flag, "Left : ", left_flag, "Right : ", right_flag)        
                time0=time.time()
                final.main2(go_flag, back_flag, left_flag, right_flag)


def send_video(connection):
        try:
                camera = cv2.VideoCapture(0)
                camera.set(3, 224)
                camera.set(4, 224)
                
                stream = io.BytesIO()
                
                while True:
                        
                        ret,frame = camera.read()
                       
                        ret,frame=cv2.imencode('.jpeg', frame)

                        image_data = frame.tobytes()
                        
                        connection.write(struct.pack('<L', len(image_data)))
                        connection.flush()
                        
                        connection.write(image_data)

                connection.write(struct.pack('<L', 0))

        finally:
            connection.close()
            client_socket.close()

dirct_thread = threading.Thread(target=receive_dirct, args=(client_socket,))
dirct_thread.start()

video_thread = threading.Thread(target=send_video, args=(connection,))
video_thread.start()

video_thread.join()
dirct_thread.join()

client_socket.close()
