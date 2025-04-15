from machine import Pin, Timer
import time

# 定义LED控制对象
led1 = Pin(23, Pin.OUT)

# 定义LED状态
led1_state = 0

# 定时器0中断函数
def time0_irq(timer):
    global led1_state
    led1_state = not led1_state  # 切换LED状态
    led1.value(led1_state)  # 设置LED的输出状态

# 程序入口
if __name__ == "__main__":
    # 创建time0定时器对象，定时周期为3000ms（3秒）
    time0 = Timer(0)
    time0.init(period=500, mode=Timer.PERIODIC, callback=time0_irq)  # 500ms周期实现1Hz的闪烁频率

    # 主程序可以不再需要`while True`循环，因为LED闪烁已由定时器控制
    # 主线程可以用于其他任务，或者直接休眠
    while True:
        time.sleep(1)  # 使主程序不退出，保持定时器工作
