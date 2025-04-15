��#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <semaphore.h>
#include <unistd.h>

#define MAX_FRUITS 5  // ��������

// �ź���
sem_t empty;  // ���̿�λ
sem_t full;   // ������ˮ������
sem_t mutex;  // �����ź���

int fruit_basket[MAX_FRUITS];  
int fruit_count = 0;            // ��ǰ����

// ��ˮ������
void put_fruit(int type) {
    fruit_basket[fruit_count] = type;
    fruit_count++;
    printf("����һ��ˮ��, ��ǰˮ��������%d\n", fruit_count);
}

// ȡˮ������
void take_fruit(int* fruit_count_local) {
    (*fruit_count_local)--;
    printf("ȡ��һ��ˮ��, ��ǰˮ��������%d\n", *fruit_count_local);
}

// �ְַ�ƻ��
void* dad(void* arg) {
    while (1) {
        sem_wait(&empty);  // �ȴ���λ
        sem_wait(&mutex);  // �ٽ���

        // ��ƻ��
        if (fruit_count < MAX_FRUITS) {
            put_fruit(1);  // 1��ʾƻ��
        }

        sem_post(&mutex);  // �뿪�ٽ���
        sem_post(&full);   // �����ѷŵ�����

        sleep(1);  // ģ���ʱ��
    }
    return NULL;
}

// ���������
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

// ȡ����
void* son(void* arg) {
    while (1) {
        sem_wait(&full);  // �ȴ�������ˮ��
        sem_wait(&mutex);  // ���ٽ���

        // ȡ����
        if (fruit_count > 0) {
            take_fruit(&fruit_count);
        }

        sem_post(&mutex);  // �뿪�ٽ���
        sem_post(&empty);  // ���ӿ�λ��

        sleep(1);  // ģ��ȡˮ����ʱ��
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

    // ��ʼ���ź���
    if (sem_init(&empty, 0, MAX_FRUITS) != 0 ||
        sem_init(&full, 0, 0) != 0 ||
        sem_init(&mutex, 0, 1) != 0) {
        perror("�ź�����ʼ��ʧ��");
        exit(EXIT_FAILURE);
    }

    // �����߳�
    if (pthread_create(&dad_thread, NULL, dad, NULL) ||
        pthread_create(&mom_thread, NULL, mom, NULL) ||
        pthread_create(&son_thread, NULL, son, NULL) ||
        pthread_create(&daughter_thread, NULL, daughter, NULL)) {
        perror("�̴߳���ʧ��");
        exit(EXIT_FAILURE);
    }

    // �ȴ��߳̽���
    pthread_join(dad_thread, NULL);
    pthread_join(mom_thread, NULL);
    pthread_join(son_thread, NULL);
    pthread_join(daughter_thread, NULL);

    // �����ź���
    sem_destroy(&empty);
    sem_destroy(&full);
    sem_destroy(&mutex);

    return 0;
}
