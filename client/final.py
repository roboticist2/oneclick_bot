import os
import RPi.GPIO as GPIO
import numpy as np
from time import sleep

GPIO.setwarnings(False)
GPIO.setmode(GPIO.BCM)
servo_pin = 12                  

GPIO.setup(servo_pin, GPIO.OUT)  
servo = GPIO.PWM(servo_pin, 50)
servo.start(0)  

servo_min_duty = 2           
servo_max_duty = 12   

RIGHT_FORWARD = 26
RIGHT_BACKWARD = 19
RIGHT_PWM = 13
LEFT_FORWARD = 16
LEFT_BACKWARD = 20
LEFT_PWM = 21

neutral_deg = 95
left_deg = 65
right_deg = 125

GPIO.setup(RIGHT_FORWARD,GPIO.OUT)
GPIO.setup(RIGHT_BACKWARD,GPIO.OUT)
GPIO.setup(RIGHT_PWM,GPIO.OUT)
GPIO.output(RIGHT_PWM,0)
RIGHT_MOTOR = GPIO.PWM(RIGHT_PWM,100)
RIGHT_MOTOR.start(0)
RIGHT_MOTOR.ChangeDutyCycle(0)
 
GPIO.setup(LEFT_FORWARD,GPIO.OUT)
GPIO.setup(LEFT_BACKWARD,GPIO.OUT)
GPIO.setup(LEFT_PWM,GPIO.OUT)
GPIO.output(LEFT_PWM,0)
LEFT_MOTOR = GPIO.PWM(LEFT_PWM,100)
LEFT_MOTOR.start(0)
LEFT_MOTOR.ChangeDutyCycle(0)

#RIGHT Motor control
def rightMotor(forward, backward, pwm):
    GPIO.output(RIGHT_FORWARD,forward)
    GPIO.output(RIGHT_BACKWARD,backward)
    RIGHT_MOTOR.ChangeDutyCycle(pwm)
 
#Left Motor control
def leftMotor(forward, backward, pwm):
    GPIO.output(LEFT_FORWARD,forward)
    GPIO.output(LEFT_BACKWARD,backward)
    LEFT_MOTOR.ChangeDutyCycle(pwm)

def motor_stop():
    GPIO.output(RIGHT_FORWARD,False)
    GPIO.output(RIGHT_BACKWARD,False)
    RIGHT_MOTOR.ChangeDutyCycle(0)
    GPIO.output(LEFT_FORWARD,False)
    GPIO.output(LEFT_BACKWARD,False)
    LEFT_MOTOR.ChangeDutyCycle(0)

def set_servo_degree(degree):    
    if degree > 180:
        degree = 170
    elif degree < 0:
        degree = 10

    duty = servo_min_duty+(degree*(servo_max_duty-servo_min_duty)/180.0)
    servo.ChangeDutyCycle(duty)                     


"""
def main():
    camera = cv2.VideoCapture(0)
    camera.set(3, 640)
    camera.set(4, 480)
    
    carState = 'stop'
    carState2 = ''

    np.set_printoptions(suppress=True)

    model = load_model("/home/pi/Desktop/keras_model.h5", compile=False)
    class_names = open("/home/pi/Desktop/labels.txt", "r").readlines()
    
    fc = 0
    while(camera.isOpened()):
        keyValue = cv2.waitKey(1)
        if keyValue == ord('q'):
            break
        if keyValue == 82:
            print('go')
            carState = 'go'
            index = 0
        if keyValue == 84:
            print('stop')
            carState = 'stop'
        if keyValue == 81:
            print('left')
            index = 1
        if keyValue == 83:
            print('right')
            index = 2
            
        ret, image = camera.read()

        image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA)

        cv2.imshow("Webcam Image", image)

        image = np.asarray(image, dtype=np.float32).reshape(1, 224, 224, 3)

        image = (image / 127.5) - 1

        if fc%6==0:
            prediction = model.predict(image)
            #prediction = np.array([[1, 0, 0]])
            index = np.argmax(prediction[0])
            class_name = class_names[index]
            confidence_score = prediction[0][index]
            print(keyValue,'  ', index, '   ', prediction[0])
        fc = fc+1
        
        #print(keyValue,'  ', index, '   ', prediction[0])
        #print("Class:", class_name[2:], end="")
        #print("Confidence Score:", str(np.round(confidence_score * 100))[:-2], "%")

        if carState == 'go':
            if index == 0 and carState2 != 'go':
                print("--------------go")
                carState2 = 'go'
                set_servo_degree(90)
                rightMotor(1 ,0, 50)
                leftMotor(1 ,0, 50)
            elif index == 2 and carState2 != 'right':
                print("--------------right")
                carState2 = 'right'
                set_servo_degree(120)
                rightMotor(1 ,0, 30)
                leftMotor(1 ,0, 50)
            elif index == 1 and carState2 != 'left':
                print("--------------left")
                carState2 = 'left'
                set_servo_degree(60)
                rightMotor(1 ,0, 50)
                leftMotor(1 ,0, 30)
        elif carState == 'stop':
            print('stop')
            carState2 = 'stop'
            motor_stop()
            
    cv2.destroyAllWindows() 

if __name__ == '__main__':
    main()
    GPIO.cleanup()
"""

def main2(go_flag, back_flag, left_flag, right_flag):
    carState2 = ''

    if go_flag == back_flag:
        motor_stop()
        if left_flag==right_flag:
            set_servo_degree(neutral_deg)
        elif left_flag == 1:
            set_servo_degree(left_deg)
        elif right_flag == 1:
            set_servo_degree(right_deg)
    elif go_flag == 1:
        if left_flag==right_flag:
            set_servo_degree(neutral_deg)
            rightMotor(1 ,0, 50)
            leftMotor(1 ,0, 50)
        elif left_flag == 1:
            set_servo_degree(left_deg)
            rightMotor(1 ,0, 30)
            leftMotor(1 ,0, 50)
        elif right_flag == 1:
            set_servo_degree(right_deg)
            rightMotor(1 ,0, 50)
            leftMotor(1 ,0, 30)
    elif back_flag == 1 :
        if left_flag==right_flag:
            set_servo_degree(neutral_deg)
            rightMotor(0 ,1, 50)
            leftMotor(0 ,1, 50)
        elif left_flag == 1:
            set_servo_degree(left_deg)
            rightMotor(0 ,1, 30)
            leftMotor(0 ,1, 50)
        elif right_flag == 1:
            set_servo_degree(right_deg)
            rightMotor(0 ,1, 50)
            leftMotor(0 ,1, 30)
    
    if go_flag == back_flag:
        motor_stop()

    
def main3():
    pass

if __name__ == '__main__':
    main()
    GPIO.cleanup()
