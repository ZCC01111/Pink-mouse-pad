# 导入所需模块
from machine import Pin
import time
import dht

# 定义DHT11控制对象，连接到GPIO27引脚
dht11 = dht.DHT11(Pin(27))

# 程序入口
if __name__ == "__main__":
    time.sleep(1)  # 首次启动间隔1秒，让传感器稳定

    while True:
        try:
            # 测量温湿度
            dht11.measure()

            # 获取温度和湿度值
            temp = dht11.temperature()
            humi = dht11.humidity()

            # 如果读取失败，temp 或 humi 为 None
            if temp is None or humi is None:
                print("DHT11传感器检测失败!")
            else:
                # 打印温湿度
                print("Temperature: {}°C  Humidity: {}%".format(temp, humi))
·
        except OSError as e:
            # 捕获并打印传感器读取错误
            print("读取传感器数据失败，错误：", e)

        # 每隔2秒读取一次数据
        time.sleep(2)
