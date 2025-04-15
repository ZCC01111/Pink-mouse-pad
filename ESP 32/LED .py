from machine import Pin
import time

# 定义按键对象，使用上拉电阻
key1 = Pin(14, Pin.PULL_UP)
key2 = Pin(27, Pin.PULL_UP)
key3 = Pin(26, Pin.PULL_UP)
key4 = Pin(25, Pin.PULL_UP)

# 定义LED控制对象
led1 = Pin(15, Pin.OUT)
led2 = Pin(2, Pin.OUT)
led3 = Pin(0, Pin.OUT)
led4 = Pin(4, Pin.OUT)

# 定义按键编号
KEY1_PRESS, KEY2_PRESS, KEY3_PRESS, KEY4_PRESS = 1, 2, 3, 4

key_en = 1  # 控制按键扫描状态

# 按键扫描函数
def key_scan():
    global key_en
    # 有按键按下
    if key_en == 1 and (key1.value() == 0 or key2.value() == 0 or key3.value() == 0 or key4.value() == 0):
        time.sleep_ms(10)  # 消抖
        key_en = 0  # 按键按下时不再重复扫描
        if key1.value() == 0:
            return KEY1_PRESS
        elif key2.value() == 0:
            return KEY2_PRESS
        elif key3.value() == 0:
            return KEY3_PRESS
        elif key4.value() == 0:
            return KEY4_PRESS
    # 无按键按下，允许下一次扫描
    elif key1.value() == 1 and key2.value() == 1 and key3.value() == 1 and key4.value() == 1:
        key_en = 1
    return 0  # 无按键按下时返回 0

# 程序入口
if __name__ == "__main__":
    key = 0
    i_led1, i_led2, i_led3, i_led4 = 0, 0, 0, 0  # 定义LED状态变量
    led1.value(i_led1)  # 初始化LED，熄灭状态
    led2.value(i_led2)
    led3.value(i_led3)
    led4.value(i_led4)

    while True:  # 主循环
        key = key_scan()  # 按键扫描

        if key == KEY1_PRESS:  # 按下按键1
            i_led1 = 1 - i_led1  # 翻转LED1状态
            led1.value(i_led1)
        elif key == KEY2_PRESS:  # 按下按键2
            i_led2 = 1 - i_led2  # 翻转LED2状态
            led2.value(i_led2)
        elif key == KEY3_PRESS:  # 按下按键3
            i_led3 = 1 - i_led3  # 翻转LED3状态
            led3.value(i_led3)
        elif key == KEY4_PRESS:  # 按下按键4
            i_led4 = 1 - i_led4  # 翻转LED4状态
            led4.value(i_led4)

        time.sleep_ms(100)  # 添加延时以避免过快轮询
