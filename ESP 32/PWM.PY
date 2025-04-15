#导入Pin模块
from machine import Pin
from machine import PWM
import time

#定义LED1控制对象
led1=PWM(Pin(15), freq=1000, duty=0)

#程序入口
if __name__=="__main__":
  duty_value=0    # 占空比值
  fx=1   # 控制PWM是增大/减小方向
  while True:
      if fx==1: # 增大占空比
          duty_value+=10
          if duty_value>1010:
              fx=0
      else:   # 减小占空比
          duty_value-=10
          if duty_value<10:
              fx=1
      led1.duty(duty_value)
      time.sleep_ms(10)