#导入Pin模块
from machine import Pin
import time
from hcsr04 import HCSR04

#定义HCSR04控制对象
hcsr04=HCSR04(trigger_pin=4, echo_pin=27)

#程序入口
if __name__=="__main__":
    
    while True:
        distance=hcsr04.distance_cm()
        print("distance=%.2fCM" %distance)
        time.sleep(0.5)