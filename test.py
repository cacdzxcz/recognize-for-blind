import pygame.mixer
import pyttsx3

def prompt(text):
    engine = pyttsx3.init()
    engine.say(text)
    # 等待语音合成完成
    engine.runAndWait()