from machine import Pin, Timer
import dht
from neopixel import NeoPixel
import time
import tm1637


# 定义RGB控制对象
rp = 33  # rp 是 RGB 灯的控制引脚编号，这里是 33
rn = 1  # rn 是 RGB 灯串联的数量，这里是 1 个
rgb = NeoPixel(Pin(rp, Pin.OUT), rn)  # 创建 NeoPixel 对象，使用 Pin 类将 rp 引脚配置为输出，控制 rn 个 RGB 灯


# 定义蜂鸣器对象
bz = Pin(32, Pin.OUT)  # bz 是蜂鸣器的引脚，将 32 号引脚配置为输出


# 定义按键对象
b1 = Pin(25, Pin.IN, Pin.PULL_UP)  # b1 是按键 K1，将 25 号引脚配置为输入，并启用上拉电阻
b2 = Pin(26, Pin.IN, Pin.PULL_UP)  # b2 是按键 K2，将 26 号引脚配置为输入，并启用上拉电阻
b3 = Pin(27, Pin.IN, Pin.PULL_UP)  # b3 是按键 K3，将 27 号引脚配置为输入，并启用上拉电阻


# 定义颜色
R = (255, 0, 0)  # 红色，用 RGB 值表示
O = (255, 165, 0)  # 橙色，用 RGB 值表示
Y = (255, 150, 0)  # 黄色，用 RGB 值表示
G = (0, 255, 0)  # 绿色，用 RGB 值表示
B = (0, 0, 255)  # 蓝色，用 RGB 值表示


# 数码管初始化
smg = tm1637.TM1637(clk=Pin(23), dio=Pin(22))  # 初始化 TM1637 数码管，clk 引脚为 23 号，dio 引脚为 22 号


# DHT11 温度传感器初始化
dht_sensor = dht.DHT11(Pin(13))  # dht_sensor 是 DHT11 温度传感器，使用 13 号引脚


# 全局变量
temp = None  # temp 用于存储读取到的温度值，初始化为 None
ct = Timer(0)  # ct 是一个定时器对象，用于倒计时
cd = 0  # cd 是当前倒计时的秒数，初始为 0
ts = "RED"  # ts 是交通灯的当前状态，初始为红色
mo = False  # mo 表示是否处于手动控制模式，初始为 False
ss = None  # ss 用于保存自动状态，初始为 None


def set_rgb(color):
    """设置RGB灯颜色"""
    for i in range(rn):  # 遍历每个 RGB 灯
        rgb[i] = color  # 将颜色赋值给当前 RGB 灯
    rgb.write()  # 将颜色信息写入 RGB 灯，使其显示相应颜色


def beep():
    """短响蜂鸣器"""
    print("蜂鸣器响")  # 打印蜂鸣器响的信息
    bz.value(1)  # 将蜂鸣器引脚置为高电平，蜂鸣器响
    time.sleep(0.1)  # 等待 0.1 秒
    bz.value(0)  # 将蜂鸣器引脚置为低电平，蜂鸣器停止响


def read_temperature(timer):
    """读取温度传感器数据"""
    global temp  # 使用全局变量 temp
    try:
        dht_sensor.measure()  # 尝试测量温度
        tc = dht_sensor.temperature()  # 获取温度值
        temp = tc  # 将温度值存储在 temp 中
        print("当前温度为：", tc)  # 打印当前温度
    except Exception as e:
        print("温度读取失败:", e)  # 打印温度读取失败信息及异常信息
        temp = None  # 如果读取失败，将 temp 置为 None


def update_countdown(timer):
    """更新倒计时显示"""
    global cd, ts, mo
    if mo:  # 如果处于手动控制模式
        return  # 不更新倒计时，直接返回
    if cd > 0:  # 如果倒计时秒数大于 0
        smg.number(cd)  # 在数码管上显示当前倒计时秒数
        cd -= 1  # 倒计时秒数减 1
    else:  # 如果倒计时结束
        ct.deinit()  # 停止倒计时定时器
        # 切换交通灯状态
        if ts == "GREEN":  # 如果当前是绿灯
            ts = "YELLOW"  # 切换为黄灯
            start_countdown(3, Y)  # 开始 3 秒的倒计时，显示黄色
        elif ts == "YELLOW":  # 如果当前是黄灯
            ts = "RED"  # 切换为红灯
            start_countdown(30, R)  # 开始 30 秒的倒计时，显示红色
        elif ts == "RED":  # 如果当前是红灯
            ts = "GREEN"  # 切换为绿灯
            start_countdown(35, G)  # 开始 35 秒的倒计时，显示绿色
        beep()  # 每次状态切换时响蜂鸣器


def start_countdown(seconds, color):
    """开始倒计时并设置RGB灯颜色"""
    global cd
    cd = seconds  # 设置倒计时秒数
    set_rgb(color)  # 设置 RGB 灯的颜色
    ct.init(period=1000, mode=Timer.PERIODIC, callback=update_countdown)  # 初始化定时器，周期为 1000 毫秒（1 秒），周期性调用 update_countdown 函数


def handle_manual_override(pin):
    """处理手动控制模式（外部中断触发）"""
    global mo, ss, ts, cd


    if pin == b1:  # 如果按下的是按键 K1
        if not mo:  # 如果当前不是手动模式
            mo = True  # 进入手动模式
            ss = (ts, cd)  # 保存当前交通灯状态和倒计时秒数
            ct.deinit()  # 暂停当前倒计时定时器
            set_rgb(G)  # 将 RGB 灯设置为绿色
            smg.number(0)  # 数码管显示 0（手动模式下不显示倒计时）
        else:  # 如果已经是手动模式
            mo = False  # 退出手动模式
            ts, cd = ss  # 恢复之前保存的交通灯状态和倒计时秒数
            start_countdown(cd, G if ts == "GREEN" else R)  # 根据恢复的状态开始倒计时


    elif pin == b2:  # 如果按下的是按键 K2
        if not mo:  # 如果当前不是手动模式
            mo = True  # 进入手动模式
            ss = (ts, cd)  # 保存当前交通灯状态和倒计时秒数
            ct.deinit()  # 暂停当前倒计时定时器
            set_rgb(O)  # 将 RGB 灯设置为橙色
            smg.number(0)  # 数码管显示 0（手动模式下不显示倒计时）
        else:  # 如果已经是手动模式
            mo = False  # 退出手动模式
            ts, cd = ss  # 恢复之前保存的交通灯状态和倒计时秒数
            start_countdown(cd, O if ts == "ORANGE" else R)  # 根据恢复的状态开始倒计时


    elif pin == b3:  # 如果按下的是按键 K3
        if not mo:  # 如果当前不是手动模式
            mo = True  # 进入手动模式
            ss = (ts, cd)  # 保存当前交通灯状态和倒计时秒数
            ct.deinit()  # 暂停当前倒计时定时器
            set_rgb(B)  # 将 RGB 灯设置为蓝色
            smg.number(0)  # 数码管显示 0（手动模式下不显示倒计时）
        else:  # 如果已经是手动模式
            mo = False  # 退出手动模式
            ts, cd = ss  # 恢复之前保存的交通灯状态和倒计时秒数
            start_countdown(cd, B if ts == "BLUE" else R)  # 根据恢复的状态开始倒计时


# 主程序
if __name__ == "__main__":
    # 初始化温度定时器，每3秒读取一次温度
    tt = Timer(-1)  # 创建一个定时器对象 tt
    # 初始化定时器，周期为 500 毫秒，周期性调用 read_temperature 函数，常量，通常用于定时器的初始化配置
    tt.init(period=500, mode=Timer.PERIODIC, callback=read_temperature)  


    # 绑定外部中断
    b1.irq(trigger=Pin.IRQ_FALLING, handler=handle_manual_override)  # 当按键 K1 按下（下降沿触发）时调用 handle_manual_override 函数
    b2.irq(trigger=Pin.IRQ_FALLING, handler=handle_manual_override)  
    b3.irq(trigger=Pin.IRQ_FALLING, handler=handle_manual_override)  


    # 启动初始倒计时
    print("系统启动")  # 打印系统启动信息
    start_countdown(30, R)  # 开始初始倒计时，时长 30 秒，初始状态为红色
