from math import sin, cos, pi
from lx16a import *
import time
import numpy as np

LX16A.initialize("/dev/tty.usbserial-110")

try:
    #INITIALIZE MOTOR INSTANCES

    #RIGHT LEG
    servo1 = LX16A(1) #right thigh
    servo2 = LX16A(2) #right shin
    servo3 = LX16A(3) #right ankle

    #LEFT LEG
    servo4 = LX16A(4) #left thigh
    servo5 = LX16A(5) #left shin
    servo6 = LX16A(6) #left ankle

#Exception handling in the event one of the motors is disconnected or not connected to begin with
except ServoTimeoutError as e:
    print(f"Servo {e.id_} is not connected. Exiting...")
    quit()

start = t.time()


wait = True
#Pause for 1 seconds before beginning sequence
while wait:
    if t.time()-start > 1:
        wait = False

print('ready')


###############################
## Initialization Position
###############################


#Right Leg
servo1.move(68, time=550)
servo2.move(105.0, time=550)
servo3.move(63, time=550)
print(servo1.get_physical_angle())
print(servo2.get_physical_angle())
print(servo3.get_physical_angle())
print('RIGHT MOTOR INITIALIZED')

time.sleep(3)

#Left Leg
servo4.move(56, time=550)
servo5.move(150, time=550)
servo6.move(70, time=550)

print(servo4.get_physical_angle())
print(servo5.get_physical_angle())
print(servo6.get_physical_angle())
print('LEFT MOTOR INITIALIZED')

time.sleep(3)

print('End of Calibration:')
print('servo 1 physical',servo1.get_physical_angle())
print('servo 2 physical',servo2.get_physical_angle())
print('servo 3 physical',servo3.get_physical_angle())
print('servo 4 physical',servo4.get_physical_angle())
print('servo 5 physical',servo5.get_physical_angle())
print('servo 6 physical',servo6.get_physical_angle())


