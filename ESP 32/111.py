'''
实验名称：点亮第一个LED灯
接线说明：LED模块--> ESP32 GPIO
连接引脚：D1-->15
         
实验现象：程序下载成功后，D1指示灯点亮

'''

#导入Pin模块
from machine import Pin


#构建led1对象，GPIO 15输出
led1=Pin(15,Pin.OUT)

# #使IO15输出高电平 --> 点亮LED
led1.value(1) 
