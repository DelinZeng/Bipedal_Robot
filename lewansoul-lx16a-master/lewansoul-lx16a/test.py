import serial
import lewansoul_lx16a

SERIAL_PORT = '/dev/tty.usbserial-110'

controller = lewansoul_lx16a.ServoController(
    serial.Serial(SERIAL_PORT, 115200, timeout=1),
)

# # control servos directly
# controller.move(1, 0) # move servo ID=1 to position 100

# or through proxy objects
servo1 = controller.servo(1)
servo2 = controller.servo(2)
servo3 = controller.servo(3)
servo4 = controller.servo(4)
servo5 = controller.servo(5)
servo6 = controller.servo(6)

# #time unit: milliseconds,
# servo1.move(300, time=1000)
# servo2.move(300, time=1000)
# servo3.move(300, time=1000)

#synchronous move of multiple servos
servo1.move_prepare(180, time = 1000)
servo2.move_prepare(180, time = 1000)
servo3.move_prepare(180, time = 1000)
servo4.move_prepare(180, time = 1000)
servo5.move_prepare(180, time = 1000)
servo6.move_prepare(180, time = 1000)
controller.move_start()