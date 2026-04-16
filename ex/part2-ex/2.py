from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os

load_dotenv()

API_KEY = os.getenv("API_KEY")
BASE_URL = os.getenv("BASE_URL")

if not API_KEY:
    raise ValueError("未检测到 API_KEY，请检查 .env 文件是否配置正确")

chat_model = ChatOpenAI(
    api_key=API_KEY,
    base_url=BASE_URL,
    model="deepseek-chat",
    temperature=0.3,
    max_tokens=200
)

# 初始化对话历史（包含 system 设定）
history = [
    {"role": "system", "content": "你是一个耐心的AI学习助手，回复简洁易懂，适合高校学生理解。"}
]

# 第一轮对话
history.append({"role": "user", "content": "请用3句话解释什么是LangChain？"})

result = chat_model.invoke(history)
print("【第一轮回复】：")
print(result.content)

# 将模型的回复添加到历史中（assistant 消息）
history.append({"role": "assistant", "content": result.content})

# 第二轮对话
# 追问，模型需要上下文才能理解"它"
history.append({"role": "user", "content": "它的核心组件有哪些？"})

result = chat_model.invoke(history)
print("\n【第二轮回复】：")
print(result.content)

# 继续记录
history.append({"role": "assistant", "content": result.content})

# 第三轮对话
history.append({"role": "user", "content": "给我一个简单的使用场景"})

result = chat_model.invoke(history)
print("\n【第三轮回复】：")
print(result.content)

