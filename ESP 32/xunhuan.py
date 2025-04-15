from machine import Pin
import time

# 假设蜂鸣器连接到 GPIO 25 引脚
beep = Pin(25, Pin.OUT)

i = 0
while True:
    i = not i
    beep.value(i)  # 控制蜂鸣器的开关
    time.sleep_ms(1)  # 每1毫秒切换一次
