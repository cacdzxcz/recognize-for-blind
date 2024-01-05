import requests
import base64
import constants
import cv2
import pygame.mixer
import os

def get_access_token():
    """
    使用 AK，SK 生成鉴权签名（Access Token）
    :return: access_token，或是None(如果错误)
    """
    API_KEY = "4V52x02UbmM6IT85AgFXGiBI"
    SECRET_KEY = "RmFjOLqGCl3Q6ylUagnYVgQvcomVT7C9"
    url = "https://aip.baidubce.com/oauth/2.0/token"
    params = {"grant_type": "client_credentials", "client_id": API_KEY, "client_secret": SECRET_KEY}
    return str(requests.post(url, params=params).json().get("access_token"))

def object_recognize(img):
    """
    # 初始化 Pygame
    pygame.init()
    # 初始化声音模块
    pygame.mixer.init()
    # 选择音频文件
    audio_file = "/voice/begin-recognize.mp3"
    # 读取音频文件
    sound = pygame.mixer.Sound(audio_file)
    # 播放音频
    sound.play()
    """
    cv2.imwrite('./save_img1.jpg', img)
    request_url = "https://aip.baidubce.com/rest/2.0/image-classify/v1/classify/ingredient"


    img = base64.b64encode(cv2.imencode('.jpg', img)[1].tostring())

    params = {"image": img}
    access_token = get_access_token()
    request_url = request_url + "?access_token=" + access_token
    headers = {'content-type': 'application/x-www-form-urlencoded'}
    response = requests.post(request_url, data=params, headers=headers)
    if response:
        return response.json()
    return None

"""
# 指定文件夹路径
folder_path = 'C:\\Users\\86157\\Desktop\\ai_hand\\data'

# 获取文件夹下所有文件
file_list = os.listdir(folder_path)

# 遍历文件列表
for file_name in file_list:
    # 拼接文件的完整路径
    file_path = os.path.join(folder_path, file_name)

    # 确保是文件而不是文件夹
    if os.path.isfile(file_path):
        # 读取图像
        image = cv2.imread(file_path)
        print("#############################")
        print(file_path)
        print(object_recognize(image))
        print("#############################")
"""



