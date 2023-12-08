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
