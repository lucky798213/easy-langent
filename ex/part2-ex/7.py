from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os

# 1. 环境与模型初始化（省略，同方案1）
load_dotenv()
API_KEY = os.getenv("API_KEY")
BASE_URL = os.getenv("BASE_URL")  # 从环境变量读取，未配置时默认为None（使用OpenAI官方地址）
llm = ChatOpenAI(
    api_key=API_KEY,
    base_url=BASE_URL,
    model="deepseek-chat",
    temperature=0.3
)

# 2. 创建 JSON 解析器（无需额外配置，默认引导模型输出 JSON）
parser = JsonOutputParser()

# 3. 构建提示模板（无需手动嵌入格式指令，解析器自动关联）
prompt = PromptTemplate(
    template="请介绍1个LangChain开发工具，输出工具名和核心功能。{format_instructions}",
    input_variables=[],
    partial_variables={"format_instructions": parser.get_format_instructions()}
)

# 4. 链式调用（LangChain ≥1.0.0 推荐方式，自动完成提示+调用+解析）
chain = prompt | llm | parser
result = chain.invoke({})  # 无输入参数，传入空字典

print("解析后的JSON（Python字典）：")
print(result)
print("获取单个字段：", result.get('tool_name', None))  # 可直接用于业务逻辑