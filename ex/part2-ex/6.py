from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
import os

# 1. 环境初始化
load_dotenv()
API_KEY = os.getenv("API_KEY")
BASE_URL = os.getenv("BASE_URL")  # 从环境变量读取，未配置时默认为None（使用OpenAI官方地址）

# 2. 初始化模型（无需支持原生结构化输出）
llm = ChatOpenAI(
    api_key=API_KEY,
    base_url=BASE_URL,
    model="deepseek-chat",
    temperature=0.3
)

# 3. 创建 StrOutputParser
# 核心作用：将 LLM 返回的 AIMessage 对象，统一转为纯字符串（str）
parser = StrOutputParser()

# 4. 链式调用：模型 → 字符串解析
#parser只返回content的内容，不包含其他字段
chain = llm | parser
result = chain.invoke("请简要介绍 LangChain 输出解析层的作用")

print("StrOutputParser 解析后的字符串：")
print(result)
print("\n解析结果类型：", type(result))  # str