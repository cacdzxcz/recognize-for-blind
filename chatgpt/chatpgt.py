import os
import sys
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import TextLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.indexes import VectorstoreIndexCreator
from langchain.indexes.vectorstore import VectorStoreIndexWrapper
from langchain.vectorstores import Chroma
import pygame.mixer
import constants
import pyttsx3
import test

def test_2(pipe):
    while True:
        if pipe.poll():
            i = pipe.recv()
            print(i)
            print("processes 2 completed")
            break

def voice(text):
    if 'result' in text.keys():
        result = text['result'][0]
        thing = "这是" + result['name'] + "。"

        test.prompt(thing)
        """
        tts = gTTS(text=thing, lang='en', slow=True)  # 设置语言，这里使用英语
        # 将 TTS 内容写入内存
        mp3_fp = io.BytesIO()
            tts.write_to_fp(mp3_fp)
            # 初始化音频模块
            pygame.mixer.init()

            # 创建音频对象
            sound = pygame.mixer.Sound(mp3_fp)
            # 播放音频
            sound.play()
        """
    else:
        text = "请重新识别"
        test.prompt(text)
        """
            tts = gTTS(text=thing, lang='en', slow=True)  # 设置语言，这里使用英语
            # 将 TTS 内容写入内存
            mp3_fp = io.BytesIO()
            tts.write_to_fp(mp3_fp)

            # 创建音频对象
            sound = pygame.mixer.Sound(mp3_fp)
            # 播放音频
            sound.play()
        """

def gpt():
    os.environ["OPENAI_API_KEY"] = constants.APIKEY

    # Enable to save to disk & reuse the model (for repeated queries on the same data)
    PERSIST = False

    query = input("Prompt: ")

    if PERSIST and os.path.exists("persist"):
        print("Reusing index...\n")
        vectorstore = Chroma(persist_directory="persist", embedding_function=OpenAIEmbeddings())
        index = VectorStoreIndexWrapper(vectorstore=vectorstore)
    else:
        loader = TextLoader("data.txt")  # Use this line if you only need data.txt
        # loader = DirectoryLoader("data/")
        if PERSIST:
            index = VectorstoreIndexCreator(vectorstore_kwargs={"persist_directory": "persist"}).from_loaders([loader])
        else:
            index = VectorstoreIndexCreator().from_loaders([loader])

    chain = ConversationalRetrievalChain.from_llm(
        llm=ChatOpenAI(model="gpt-3.5-turbo"),
        retriever=index.vectorstore.as_retriever(search_kwargs={"k": 1}),
    )

    chat_history = []

    while True:
        if not query:
            query = input("Prompt: ")
        if query in ['quit', 'q', 'exit']:
            sys.exit()
        result = chain({"question": query, "chat_history": chat_history})
        print(result['answer'])
        """
        chat_history.append((query, result['answer']))

        # 使用 split() 函数按空格分割文本
        words = query.split()

        thing = "this is" + words[2] + "."
        answer_text = thing + result['answer']
        tts = gTTS(text=answer_text, lang='en', slow=True)  # 设置语言，这里使用英语
        tts.save("answer.mp3")
        # 初始化音频模块
        pygame.mixer.init()

        # 创建音频对象
        sound = pygame.mixer.Sound("answer.mp3")
        # 播放音频
        sound.play()
        # 播放持续时间（以毫秒为单位）
        os.remove("answer.mp3")
        """
        query = None

