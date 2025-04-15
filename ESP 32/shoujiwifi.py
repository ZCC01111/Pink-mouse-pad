from machine import Pin
import time
import network
import socket

# 控制 LED 的引脚
led1 = Pin(15, Pin.OUT, Pin.PULL_DOWN)

# Wi-Fi 连接信息
ssid = 'iQOO Neo8'  # 替换为你的Wi-Fi名称
password = '12345678'  # 替换为你的Wi-Fi密码

# 连接 Wi-Fi 的函数
def wifi_connect():
    wlan = network.WLAN(network.STA_IF)  # STA模式
    wlan.active(True)  # 激活Wi-Fi接口
    print(wlan.scan())  # 扫描可用网络
    start_time = time.time()  # 记录连接开始时间
    if not wlan.isconnected():
        print("Connecting to network...")
        wlan.connect(ssid, password)  # 连接指定的Wi-Fi
        while not wlan.isconnected():
            led1.value(1)  # 连接时闪烁 LED
            time.sleep_ms(300)
            led1.value(0)
            time.sleep_ms(300)

            if time.time() - start_time > 15:
                print("Wi-Fi Connect Timeout!")
                break

        led1.value(0)  # 连接完成后关闭 LED
        return False
    else:
        led1.value(0)
        print("Network information:", wlan.ifconfig())
        return True

# 网页内容的函数
def web_page():
    if led1.value() == 0:
        gpio_state = "OFF"
    else:
        gpio_state = "ON"

    html = """<html><head> <title>ESP32 LED control</title> 
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <link rel="icon" href="data:,"> <style>html{font-family: Helvetica; display:inline-block; 
    margin: 0px auto; text-align: center;} h1{color: #0F3376; padding: 2vh;} 
    p{font-size: 1.5rem;} .button{display: inline-block; background-color: #e7bd3b; border: none; 
    border-radius: 4px; color: white; padding: 16px 40px; text-decoration: none; font-size: 30px; 
    margin: 2px; cursor: pointer;} .button2{background-color: #4286f4;}</style></head><body> 
    <h1>ESP32 LED control</h1> <p>GPIO state: <strong>""" + gpio_state + """</strong></p>
    <p><a href="/?led=on"><button class="button">ON</button></a></p>
    <p><a href="/?led=off"><button class="button button2">OFF</button></a></p></body></html>"""
    
    return html

# 主程序
if __name__ == "__main__":
    if wifi_connect():  # 连接Wi-Fi
        my_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)  # 创建 TCP socket
        my_socket.bind(('192.168.7.193', 80))  # 绑定本机IP和端口
        my_socket.listen(5)  # 设置监听队列

        while True:
            client, addr = my_socket.accept()  # 等待客户端连接
            print('Got a connection from %s' % str(addr))
            request = client.recv(1024)  # 接收请求数据
            request = str(request)
            print('Content = %s' % request)
            
            # 判断请求路径，控制 LED 状态
            led_on = request.find('/?led=on')
            led_off = request.find('/?led=off')
            if led_on == 6:
                print('LED ON')
                led1.value(1)  # 打开 LED
            if led_off == 6:
                print('LED OFF')
                led1.value(0)  # 关闭 LED

            response = web_page()  # 获取网页内容
            client.send('HTTP/1.1 200 OK\n')
            client.send('Content-Type: text/html\n')
            client.send('Connection: close\n\n')
            client.sendall(response)  # 发送网页内容给客户端
            client.close()  # 关闭客户端连接
