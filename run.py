import random
import cv2
from hand.handpose import hand_trap
from hand.handpose import judge_index_finger_stable
from hand.handpose import plot_charging_ring
from hand.handpose import plot_key_point
from hand.handpose import img_split
from recognize_image.baidu_cloud import object_recognize
from chatgpt.chatpgt import voice
import test
import threading
import numpy as np

def test_1(pipe):
    while True:
        i = random.randint(1, 100) % 10
        print(i)
        if i == 5:
            pipe.send(i)
            print("processes 1 completed")
            break


def data_augmentation(image):
    # 随机水平翻转
    if np.random.rand() < 0.5:
        image = cv2.flip(image, 1)

    # 随机垂直翻转
    if np.random.rand() < 0.5:
        image = cv2.flip(image, 0)

    # 随机旋转
    angle = np.random.randint(-30, 30)
    rows, cols, _ = image.shape
    M = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)
    image = cv2.warpAffine(image, M, (cols, rows))

    # 随机调整亮度和对比度
    alpha = 1.0 + np.random.uniform(-0.2, 0.2)
    beta = np.random.uniform(-20, 20)
    image = cv2.addWeighted(image, alpha, np.zeros_like(image), 0, beta)

    return image


def make_reco(img):
    prompt = "开始识别。"
    test.prompt(prompt)
    result = object_recognize(img)
    print(result)
    voice(result)

def video():
    # 通过局域网传递数据
    video = "http://admin:admin@192.168.43.214:8081"
    # video = "http://admin:admin@172.25.190.175:8081"
    # video = "http://admin:admin@[2001:250:5800:1002::8b:3384]:8081"
    cap = cv2.VideoCapture(1)
    print("start handpose process ~")
    window_name = 'MediaPipe Hand Tracking'
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)  # 使用cv2.WINDOW_NORMAL标志以允许手动调整窗口大小
    cv2.resizeWindow(window_name, 900, 700)  # 设置窗口大小

    # 记录上一帧图像中全部的手并分配id，手部跟踪      all_hands[hand_id] = [detected_hand_bbox, last_frame, index_finger_site, landmarks]
    all_hands = {}
    # 记录上一帧图像中全部的食指的位置，并分配id      all_index_finger[hand_id] = [current_index_finger_site, last_frame]
    all_index_finger = {}
    # 稳定的食指的数量                           stable_index_finger[finger_id] = bool
    stable_index_finger = {}

    hands_num_change = 0

    while cap.isOpened():
        # 读取视频帧
        ret, frame = cap.read()
        if not ret:
            break

        img_hand = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # 处理帧
        # 手部识别以及手部跟踪
        all_hands = hand_trap(img_hand, all_hands)

        # 手部数量变化时提示
        if hands_num_change != len(all_hands):
            hands_num_change = len(all_hands)
            if hands_num_change != 0:
                text_prompt = "识别两只手。"
            else:
                text_prompt = "识别零只手。"
            prompt_thread = threading.Thread(target=test.prompt, args=(text_prompt,))
            prompt_thread.start()

        # 手部食指指尖跟踪
        all_index_finger = judge_index_finger_stable(all_hands, all_index_finger)
        # 绘制手部关键点
        frame = plot_key_point(all_hands, frame)
        # 绘制充电环
        frame, stable_index_finger = plot_charging_ring(all_index_finger, frame)
        # 确定识别物体，扣出图像
        img, frame = img_split(stable_index_finger, all_index_finger, frame)
        if not img is None:
            reco_thread = threading.Thread(target=make_reco, args=(img,))
            reco_thread.start()
        # 显示帧
        cv2.imshow(window_name, frame)

        # 按 'q' 键退出循环
        if cv2.waitKey(1) & 0xFF == ord('q'):
            result = "停止识别。"
            prompt_thread = threading.Thread(target=test.prompt, args=(result,))
            prompt_thread.start()
            break

    # 释放资源
    cap.release()
    cv2.destroyAllWindows()

"""
# 初始化 Pygame
pygame.init()
# 初始化声音模块
pygame.mixer.init()
# 选择音频文件
audio_file = "/voice/begin.mp3"
# 读取音频文件
sound = pygame.mixer.Sound(audio_file)
# 播放音频
sound.play()
"""
video()
"""
# 创建两个进程，分别运行实时视频与实时语音进程
if __name__ == "__main__":
    multiprocessing.freeze_support()
    parent_pipe, child_pipe = multiprocessing.Pipe()

    # 创建进程，并将Queue传递给每个进程
    process1 = multiprocessing.Process(target=video, args=(parent_pipe, ))
    process2 = multiprocessing.Process(target=voice, args=(child_pipe,))

    # 启动进程
    process1.start()
    process2.start()

    # 等待两个进程完成
    process1.join()
    process2.join()

    print("All processes completed")
"""




