"""
import socket
client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client_socket.connect(('192.168.43.156', 9999))

message = "Hello, Server!"
client_socket.send(message.encode())

client_socket.close()

# ip: 172.25.204.214
"""

import pyttsx3

engine = pyttsx3.init()

text = "请重新识别"

# 将文本传递给引擎
engine.say(text)

# 保存为音频文件
engine.save_to_file(text, 're-recognize.mp3')

# 等待语音合成完成
engine.runAndWait()
