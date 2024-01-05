import mediapipe as mp
import math
import matplotlib.pyplot as plt
import numpy as np
import cv2

# 初始化绘图工具
mp_drawing = mp.solutions.drawing_utils
# 初始化MediaPipe手部检测以及关键点模型
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()

# 手部跟踪，同一时间只能跟踪两个手部
def hand_trap(frame, all_hands, iou_threshold=0.5):
    # 首先进行手部检测
    results = hands.process(frame)

    if not all_hands:
        if results.multi_hand_landmarks:
            for landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, landmarks, mp_hands.HAND_CONNECTIONS)
                # 获取检测到的手的位置
                x_min = min(landmark.x for landmark in landmarks.landmark)
                y_min = min(landmark.y for landmark in landmarks.landmark)
                x_max = max(landmark.x for landmark in landmarks.landmark)
                y_max = max(landmark.y for landmark in landmarks.landmark)
                detected_hand_bbox = [x_min, y_min, x_max, y_max]

                # 获取食指的位置
                index_finger_landmark = landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                index_finger_x, index_finger_y = int(index_finger_landmark.x * frame.shape[1]), int(
                    index_finger_landmark.y * frame.shape[0])
                index_finger_site = [index_finger_x, index_finger_y]

                num_all_hands = len(all_hands)
                all_hands[num_all_hands] = [detected_hand_bbox, 1, index_finger_site, landmarks]


        return all_hands

    if not results.multi_hand_landmarks:
        # 清除上一帧所有的手
        all_hands = {}
        return all_hands
    else:
        all_hands_now = {}
        num_all_hands = len(all_hands)
        for landmarks in results.multi_hand_landmarks:
            # 获取检测到的手的位置
            x_min = min(landmark.x for landmark in landmarks.landmark)
            y_min = min(landmark.y for landmark in landmarks.landmark)
            x_max = max(landmark.x for landmark in landmarks.landmark)
            y_max = max(landmark.y for landmark in landmarks.landmark)
            detected_hand_bbox = [x_min, y_min, x_max, y_max]
            # 获取食指的位置
            index_finger_landmark = landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            index_finger_x, index_finger_y = int(index_finger_landmark.x * frame.shape[1]), int(
                index_finger_landmark.y * frame.shape[0])
            index_finger_site = [index_finger_x, index_finger_y]

            # 判断是否是新的手部
            new_hand = True
            for hand_id in all_hands.keys():
                # 首先获取上一帧中出现的某一只手的位置
                hand = all_hands[hand_id]
                expected_hand_bbox = hand[0]

                # 计算IoU
                intersection_area = max(0, min(expected_hand_bbox[2], detected_hand_bbox[2]) - max(expected_hand_bbox[0], detected_hand_bbox[0])) * \
                                    max(0, min(expected_hand_bbox[3], detected_hand_bbox[3]) - max(expected_hand_bbox[1], detected_hand_bbox[1]))

                expected_area = (expected_hand_bbox[2] - expected_hand_bbox[0]) * (expected_hand_bbox[3] - expected_hand_bbox[1])
                detected_area = (detected_hand_bbox[2] - detected_hand_bbox[0]) * (detected_hand_bbox[3] - detected_hand_bbox[1])

                iou = intersection_area / (expected_area + detected_area - intersection_area)


                # 如果前后两帧之间多个手靠的近，选择最先遍历到的那个
                # 如果IoU满足设定阈值，更新手部信息以及食指指尖信息
                if iou > iou_threshold:
                    last_frame_hand = all_hands[hand_id][1] + 1
                    all_hands_now[hand_id] = [detected_hand_bbox, last_frame_hand, index_finger_site, landmarks]
                    new_hand = False
                    break
            if new_hand:
                # 如果没有，则添加新的手部
                all_hands_now[num_all_hands] = [detected_hand_bbox, 1, index_finger_site, landmarks]
                num_all_hands = num_all_hands + 1

        return all_hands_now

# 判断食指状态是否稳定
def judge_index_finger_stable(all_hands, all_index_finger, distance_threshold=10):
    if not all_index_finger:
        for hand_id in all_hands.keys():
            all_index_finger[hand_id] = [all_hands[hand_id][2], 1]
        return all_index_finger

    if not all_hands:
        all_index_finger = {}
        return all_index_finger
    else:
        for hand_id in all_hands.keys():
            if hand_id not in all_index_finger:
                # new index finger
                all_index_finger[hand_id] = [all_hands[hand_id][2], 1]
            else:
                # 上一帧的食指指尖位置以及位置稳定的帧数
                last_index_finger_site = all_index_finger[hand_id][0]
                last_frame = all_index_finger[hand_id][1]

                current_index_finger_site = all_hands[hand_id][2]
                distance = math.sqrt((current_index_finger_site[0] - last_index_finger_site[0])**2 + (current_index_finger_site[1] - last_index_finger_site[1])**2)

                # 当前食指稳定
                if distance <= distance_threshold:
                    all_index_finger[hand_id] = [current_index_finger_site, last_frame + 1]
                else:
                    all_index_finger[hand_id] = [current_index_finger_site, 1]
        return all_index_finger


def plot_key_point(all_hands, frame):
    for hand_id in all_hands.keys():
        landmarks = all_hands[hand_id][3]
        mp_drawing.draw_landmarks(frame, landmarks, mp_hands.HAND_CONNECTIONS)
    return frame

# 绘制充电环
def plot_charging_ring(all_index_finger, frame, charge_cycle_step=30, frame_threshold=40):
    stable_index_finger = {}

    for index_finger_id in all_index_finger:
        last_frame = all_index_finger[index_finger_id][1]
        index_finger_site = all_index_finger[index_finger_id][0]

        if last_frame > frame_threshold:
            fill_cnt = int((last_frame - frame_threshold) * charge_cycle_step)
            if fill_cnt < 360:
                cv2.ellipse(frame, index_finger_site, (16, 16), 0, 0, fill_cnt, (255, 255, 0), 2)
            else:
                cv2.ellipse(frame, index_finger_site, (16, 16), 0, 0, fill_cnt, (0, 150, 255), 4)
                stable_index_finger[index_finger_id] = True
    return frame, stable_index_finger


def img_split(stable_index_finger, all_index_finger, frame):
    if len(stable_index_finger) == 2:

        # 将两个食指指尖存在的帧数清零
        for finger_id in stable_index_finger.keys():
            all_index_finger[finger_id][1] = 0

        points = []
        for index_finger_stable_id in stable_index_finger:
            points.append(all_index_finger[index_finger_stable_id][0])

        x1, y1 = int(points[0][0]), int(points[0][1])
        x2, y2 = int(points[1][0]), int(points[1][1])
        if x2 < x1:
            x1, x2 = x2, x1
        if y2 < y1:
            y1, y2 = y2, y1

        img = frame[y1:y2, x1:x2]

        # 将图像贴在原始帧的右下角
        height, width, _ = frame.shape
        img_height, img_width, _ = img.shape

        # 计算贴图位置
        y_offset = height - img_height
        x_offset = width - img_width


        # 将图像贴在原始帧上
        frame[y_offset:height, x_offset:width] = img
        return img, frame
    return None, frame











