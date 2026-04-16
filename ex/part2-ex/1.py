# 导入必要的模块
from langchain_openai import ChatOpenAI  # OpenAI对话模型的统一接口
from dotenv import load_dotenv
import os

# 加载API密钥（和上一章一样，从.env文件读取）
load_dotenv()

API_KEY = os.getenv("API_KEY")
BASE_URL = os.getenv("BASE_URL")  # 从环境变量读取，未配置时默认为None（使用OpenAI官方地址）

if not API_KEY:
    raise ValueError("未检测到 API_KEY，请检查 .env 文件是否配置正确")

# 1. 初始化对话模型
# 不管是哪个厂商的ChatModel，初始化参数都类似（model、temperature等）
chat_model = ChatOpenAI(
    api_key=API_KEY,
    base_url=BASE_URL,
    model="deepseek-chat",  # 选择对话模型
    temperature=0.3,        # 随机性：0-1，越小越严谨，越大越有创造力
    max_tokens=200          # 最大生成 tokens 数，避免生成过长内容
)

# 2. 构造对话消息
# ChatModel需要接收的是“消息列表”，每个消息有角色（user/assistant/system）和内容
messages = [
    # system消息：给助手设定身份和行为准则，会影响后续所有回复
    {"role": "system", "content": "你是一个耐心的AI学习助手，回复简洁易懂，适合高校学生理解。"},
    # user消息：用户的问题
    {"role": "user", "content": "请用3句话解释什么是LangChain？"}
]

# 3. 调用模型生成结果
# 统一调用方法：invoke()，传入消息列表
result = chat_model.invoke(messages)

# 4. 输出结果
# 结果是一个ChatMessage对象，content属性是回复内容
print("ChatModel回复：")
print(result.content)