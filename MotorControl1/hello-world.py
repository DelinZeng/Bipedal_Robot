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
#####DEFAULT POSITIONS!!!!#####
###############################

#Right Leg
servo1.move(68, time=750, wait_to_complete=False)
servo2.move(105.0, time=750, wait_to_complete=False)
servo3.move(63, time=750, wait_to_complete=False)
print('RIGHT MOTOR INITIALIZED')

#Left Leg
servo4.move(56, time=750, wait_to_complete=False)
servo5.move(150, time=750, wait_to_complete=False)
servo6.move(70, time=750, wait_to_complete=False)

print('LEFT MOTOR INITIALIZED')

time.sleep(2)

# print(servo1.get_physical_angle())
# print(servo2.get_physical_angle())
# print(servo3.get_physical_angle())
# print(servo4.get_physical_angle())
# print(servo5.get_physical_angle())
# print(servo6.get_physical_angle())


# time.sleep(2)

###############################
##### Move up both leg #####
###############################

# #Left Leg
# servo4.move(3, time=750, wait_to_complete=False)
# servo5.move(98, time=750, wait_to_complete=False)
# servo6.move(72, time=750, wait_to_complete=False)



