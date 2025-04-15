、#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <semaphore.h>
#include <unistd.h>

#define MAX_FRUITS 5  // 果盘容量

// 信号量
sem_t empty;  // 果盘空位
sem_t full;   // 果盘中水果数量
sem_t mutex;  // 互斥信号量

int fruit_basket[MAX_FRUITS];  
int fruit_count = 0;            // 当前数量

// 放水果函数
void put_fruit(int type) {
    fruit_basket[fruit_count] = type;
    fruit_count++;
    printf("放了一个水果, 当前水果数量：%d\n", fruit_count);
}

// 取水果函数
void take_fruit(int* fruit_count_local) {
    (*fruit_count_local)--;
    printf("取了一种水果, 当前水果数量：%d\n", *fruit_count_local);
}

// 爸爸放苹果
void* dad(void* arg) {
    while (1) {
        sem_wait(&empty);  // 等待空位
        sem_wait(&mutex);  // 临界区

        // 放苹果
        if (fruit_count < MAX_FRUITS) {
            put_fruit(1);  // 1表示苹果
        }

        sem_post(&mutex);  // 离开临界区
        sem_post(&full);   // 增加已放的数量

        sleep(1);  // 模拟放时间
    }
    return NULL;
}

// 妈妈放橘子
void* mom(void* arg) {
    while (1) {
        sem_wait(&empty);  
        sem_wait(&mutex);  


        if (fruit_count < MAX_FRUITS) {
            put_fruit(2); 
        }

        sem_post(&mutex);  
        sem_post(&full);  

        sleep(1); 
    }
    return NULL;
}

// 取橘子
void* son(void* arg) {
    while (1) {
        sem_wait(&full);  // 等待果盘有水果
        sem_wait(&mutex);  // 进临界区

        // 取橘子
        if (fruit_count > 0) {
            take_fruit(&fruit_count);
        }

        sem_post(&mutex);  // 离开临界区
        sem_post(&empty);  // 增加空位数

        sleep(1);  // 模拟取水果的时间
    }
    return NULL;
}


void* daughter(void* arg) {
    while (1) {
        sem_wait(&full);  
        sem_wait(&mutex); 


        if (fruit_count > 0) {
            take_fruit(&fruit_count);
        }

        sem_post(&mutex); 
        sem_post(&empty);  

        sleep(1);  
    }
    return NULL;
}

int main() {
    pthread_t dad_thread, mom_thread, son_thread, daughter_thread;

    // 初始化信号量
    if (sem_init(&empty, 0, MAX_FRUITS) != 0 ||
        sem_init(&full, 0, 0) != 0 ||
        sem_init(&mutex, 0, 1) != 0) {
        perror("信号量初始化失败");
        exit(EXIT_FAILURE);
    }

    // 创建线程
    if (pthread_create(&dad_thread, NULL, dad, NULL) ||
        pthread_create(&mom_thread, NULL, mom, NULL) ||
        pthread_create(&son_thread, NULL, son, NULL) ||
        pthread_create(&daughter_thread, NULL, daughter, NULL)) {
        perror("线程创建失败");
        exit(EXIT_FAILURE);
    }

    // 等待线程结束
    pthread_join(dad_thread, NULL);
    pthread_join(mom_thread, NULL);
    pthread_join(son_thread, NULL);
    pthread_join(daughter_thread, NULL);

    // 销毁信号量
    sem_destroy(&empty);
    sem_destroy(&full);
    sem_destroy(&mutex);

    return 0;
}
