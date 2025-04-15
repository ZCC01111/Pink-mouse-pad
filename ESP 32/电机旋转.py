from machine import Pin
import time

# 修改拼写错误
dc_motor = Pin(15, Pin.OUT)  # Pin 15 用作输出引脚，具体电路决定是否需要使用 PULL_UP 或 PULL_DOWN

if __name__ == "__main__":
    dc_motor.value(1)  # 启动电机（高电平）
    time.sleep(3)      # 等待 3 秒
    dc_motor.value(0)  # 停止电机（低电平）
