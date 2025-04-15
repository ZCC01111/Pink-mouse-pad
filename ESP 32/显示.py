'''
实验名称：数码管显示实验
接线说明：数码管模块-->ESP32 IO
         CLK-->(16)
         DIO-->(17)
         
实验现象：程序下载成功后，数码管间隔1s从0开始计数显示
'''

#导入Pin模块
from machine import Pin
import time
import tm1637

#定义数码管控制对象
smg=tm1637.TM1637(clk=Pin(16),dio=Pin(17)) 

#程序入口
if __name__=="__main__":
    #smg.numbers(1,24)  #显示小数01.24
    #smg.hex(123)  #将十进制数转换十六进制显示
    #smg.brightness(0)  #亮度调节
    #smg.temperature(25)  #显示带温度符号°C，整数温度值
    #smg.show("1314")  #字符串显示，显示整数
    #smg.scroll("1314-520",500)  #字符串滚动显示，速度调节
    #time.sleep(5)
    n=0
    while True:
        smg.number(n)
        n+=1
        time.sleep(1)