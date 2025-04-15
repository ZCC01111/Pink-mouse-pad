from machine import Pin
import time

beep = Pin(25, Pin.OUT)

if __name__ == "__main__":
    i = 0
    while True:
        i = not i
        beep.value(i)  # 根据 i 的值控制蜂鸣器开关
        time.sleep_us(250)
