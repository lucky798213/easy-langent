"""
修改LangChain案例的Prompt，把“AI学习建议”改成“LangChain学习建议”，观察生成结果；
"""
# 1. 导入模块
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

# 2. 加载 .env 环境变量
load_dotenv()

# 3. 配置 API Key
API_KEY = os.getenv("API_KEY")
BASE_URL = os.getenv("BASE_URL")

if not API_KEY:
    raise ValueError("未检测到 API_KEY，请检查 .env 文件是否配置正确")

# 4. 初始化大模型
llm = ChatOpenAI(
    api_key=API_KEY,
    base_url=BASE_URL,
    model="deepseek-chat",  # 注意：根据你使用的模型修改名称！！！！ 后面章节不再继续说明
    temperature=0.3
)

# 5. 构造 Prompt（教学阶段用字符串更直观）
prompt = "请写一段50字左右的LangChain学习建议，语言简洁、实用，适合初学者。"

# 6. 调用模型
response = llm.invoke(prompt)

# 7. 输出结果
print("生成的学习建议：")
print(response.content)
'''
运行结果实例：
生成的学习建议：
1. **先掌握核心概念**：理解LCEL、Chain、Agent等基础组件，动手跑通官方示例。
2. **边做边学**：从RAG或智能助手等小项目开始，逐步拆解调试代码。
3. **关注生态更新**：LangChain迭代快，多查阅最新文档和社区案例。
（注：建议配合官方Tutorial实践，遇到问题优先查阅API文档与GitHub讨论。）
'''