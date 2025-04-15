from machine import Pin
import time
import network
import usocket

# 定义LED控制对象
led1 = Pin(15, Pin.OUT)

# 路由器WIFI账号和密码
ssid = "iQOO Neo8"
password = "12345678"

dest_ip = "192.168.7.195"
dest_port = 1111

# WIFI连接
def wifi_connect():
    wlan = network.WLAN(network.STA_IF)  # STA模式
    wlan.active(True)  # 激活
    start_time = time.time()  # 记录时间做超时判断
    
    if not wlan.isconnected():
        print("Connecting to network...")
        wlan.connect(ssid, password)  # 输入 WIFI 账号密码
        
        while not wlan.isconnected():
            led1.value(1)
            time.sleep_ms(300)
            led1.value(0)
            time.sleep_ms(300)
            
            # 超时判断，15秒没连接成功判定为超时
            if time.time() - start_time > 15:
                print("WIFI Connect Timeout!")
                break
    
    else:
        led1.value(0)
        print("Network information:", wlan.ifconfig())
        return wlan  # 返回WIFI连接对象，以便后续使用

# 程序入口
if __name__ == "__main__":
    wlan = wifi_connect()
    
    if wlan and wlan.isconnected():  # 如果成功连接到WiFi
        # 创建Socket连接
        sock = usocket.socket()
        addr = (dest_ip, dest_port)
        sock.connect(addr)
        sock.send("HELLO HAPI")
        
        try:
            while True:
                text = sock.recv(128)
                if text is None:
                    pass
                else:
                    print(text)
                    sock.send("I get: " + text.decode("utf-8"))
                time.sleep_ms(300)
        finally:
            sock.close()  # 确保在退出时关闭Socket连接
    else:
        print("WiFi连接失败，无法进行Socket通信。")