from machine import Pin
import time

led_Pin = [15, 2, 0, 4, 16, 17, 5, 18]
leds = []

# 创建8个 LED 引脚
for i in range(8):
    leds.append(Pin(led_Pin[i], Pin.OUT))

# 程序入口
if __name__ == "__main__":
    # 关闭所有 LED
    for n in range(8):
        leds[n].value(0)
        
    while True:
        # 点亮 LED
        for n in range(8):
            leds[n].value(1)
            time.sleep(0.06)
            
        # 熄灭 LED
        for n in range(8):
            leds[n].value(0)
            time.sleep(0.05)
