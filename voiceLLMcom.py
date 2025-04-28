import openai
from httpx_socks import SyncProxyTransport
import httpx
import os
import rospy
from std_msgs.msg import String
import sounddevice as sd
from scipy.io.wavfile import write
import speech_recognition as sr
from pynput import keyboard
import numpy as np

# 百炼API配置
# 设置代理（如果需要）
transport = SyncProxyTransport.from_url('socks5://127.0.0.1:7890')
client = httpx.Client(transport=transport)

openai_client = openai.OpenAI(
    # 若没有配置环境变量，请用百炼API Key将下行替换为：api_key="sk-xxx",
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    http_client=client
)

# 设计提示词
prompt_template = """
你是一个非常专业的机械臂控制工程师，负责将自然语言指令转换为机械臂控制按键。并且你的输出结果只有纯粹的字母，没有任何多余的文字
按键映射如下：
- 'w': 机械臂向前移动
- 'a': 机械臂向右移动
- 'r': 机械臂向上移动
- 'q': 机械臂夹子向左转
- 'g': 机械臂俯仰增加
- 'z': 夹爪打开
- 's': 机械臂向后移动
- 'd': 机械臂向左移动
- 'f': 机械臂向下移动
- 'e': 机械臂夹子向右转
- 't': 机械臂俯仰减少
- 'c': 夹爪关闭

请根据以下自然语言指令输出对应的按键：
{input}
"""


def get_key_from_model(question):
    """调用百炼模型将自然语言转换为按键"""
    prompt = prompt_template.format(input=question)
    try:
        response = openai_client.chat.completions.create(
            model="qwen-plus",
            messages=[
                {'role': 'system', 'content': 'You are a helpful assistant.'},
                {'role': 'user', 'content': prompt}
            ]
        )
        # 提取有效的按键信息
        key = response.choices[0].message.content.strip()
        # 定义需要移除的文本列表
        remove_texts = [
            "根据指令“打开夹子”对应的按键映射是：",
            "根据指令“闭合夹子”，对应的按键是：",
            "根据按键映射，打开夹爪对应的按键是：",
            "因此指令转换结果为："
        ]
        for text in remove_texts:
            if key.startswith(text):
                key = key.replace(text, "").strip()
        if key.startswith("`") and key.endswith("`"):
            key = key[1:-1].strip()
        # 只保留字母
        key = ''.join(filter(str.isalpha, key))
        return key
    except Exception as e:
        print(f"百炼API调用失败：{str(e)}")
        return None


def record_audio():
    print("开始录音... 按 'q' 键停止")
    frames = []
    recording = True

    def on_press(key):
        nonlocal recording
        if key.char == 'q':
            recording = False

    listener = keyboard.Listener(on_press=on_press)
    listener.start()

    try:
        with sd.InputStream(
                samplerate=44100,
                channels=1,
                dtype=np.int16,
                blocksize=1024
        ) as stream:
            while recording:
                data, _ = stream.read(1024)
                frames.append(data)
    except KeyboardInterrupt:
        recording = False

    listener.stop()

    if frames:
        data = np.concatenate(frames, axis=0)
        os.makedirs(os.path.dirname("/home/zzq/voice_qwen/outputs/temp_audio.wav"), exist_ok=True)
        write("/home/zzq/voice_qwen/outputs/temp_audio.wav", 44100, data)
        print(f"录音保存为：/home/zzq/voice_qwen/outputs/temp_audio.wav")
        return "/home/zzq/voice_qwen/outputs/temp_audio.wav"
    else:
        print("未录制到音频数据")
        return None


def recognize_speech(audio_file):
    recognizer = sr.Recognizer()
    with sr.AudioFile(audio_file) as source:
        audio_data = recognizer.record(source)
    try:
        text = recognizer.recognize_google(audio_data, language="zh-CN")
        print("语音识别结果：", text)
        return text
    except sr.UnknownValueError:
        print("语音识别失败：无法识别音频内容")
        return None
    except sr.RequestError as e:
        print(f"语音识别失败：{e}")
        return None


def main():
    rospy.init_node('robot_key_publisher', anonymous=True)
    pub = rospy.Publisher('robot_key_command', String, queue_size=10)
    rate = rospy.Rate(1)  # 1 Hz

    while not rospy.is_shutdown():
        print("请开始语音输入，按 'q' 键结束录音（输入'exit'可退出程序）...")
        audio_file = record_audio()
        if not audio_file:
            continue

        user_input = recognize_speech(audio_file)
        if user_input:
            if user_input.lower() == 'exit':
                print("接收到退出指令，程序即将退出...")
                break
            print(f"\n你输入的指令：{user_input}")
            key = get_key_from_model(user_input)

            if key:
                pub.publish(key)
                print(f"已发布按键: {key}")
            else:
                print("未能从模型获取有效按键")
        else:
            print("未成功识别语音内容，请重试。")

        rate.sleep()


if __name__ == "__main__":
    main()
