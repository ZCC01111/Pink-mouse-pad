#-*- coding: utf-8 -*-
import threading
import time

# 初始化
empty_slots = threading.Semaphore(40)  # 果盘中空数量，最多40个
apple_in_basket = threading.Semaphore(0)  # 果盘中没有苹果
orange_in_basket = threading.Semaphore(0)  # 果盘中没有橘子
mutex = threading.Semaphore(1)  # 互斥

# 爸爸不停放苹果
def dad():
    while True:
        empty_slots.acquire()  # 确保果盘中
        mutex.acquire()  # 互斥
        print("爸爸放一个苹果")
        apple_in_basket.release()  # 放入
        mutex.release()  # 释放
        time.sleep(1)

# 妈妈不停放橘子
def mom():
    while True:
        empty_slots.acquire()
        mutex.acquire()  # 互斥
        print("妈妈放一个橘子")
        orange_in_basket.release()  # 放入
        mutex.release()  # 释放
        time.sleep(1)

# 儿子不停取橘子
def son():
    while True:
        orange_in_basket.acquire()
        mutex.acquire()
        print("儿子取一个橘子")
        empty_slots.release()
        mutex.release()
        time.sleep(1)

# 女儿的线程：不停取苹果
def daughter():
    while True:
        apple_in_basket.acquire()
        mutex.acquire()
        print("女儿一个苹果")
        empty_slots.release()
        mutex.release()
        time.sleep(1)

# 创建线程
dad_thread = threading.Thread(target=dad)
mom_thread = threading.Thread(target=mom)
son_thread = threading.Thread(target=son)
daughter_thread = threading.Thread(target=daughter)


dad_thread.start()
mom_thread.start()
son_thread.start()
daughter_thread.start()

dad_thread.join()
mom_thread.join()
son_thread.join()
daughter_thread.join()
