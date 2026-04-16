from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from dotenv import load_dotenv
import os

# 1. 环境与模型初始化
load_dotenv()
API_KEY = os.getenv("API_KEY")
BASE_URL = os.getenv("BASE_URL")  # 从环境变量读取，未配置时默认为None（使用OpenAI官方地址）

llm = ChatOpenAI(
    api_key=API_KEY,
    base_url=BASE_URL,
    model="deepseek-chat",
    temperature=0.3
)

# 2. 定义 Pydantic 数据模型
class ToolInfo(BaseModel):
    #tool_name: str：声明一个字段，名字叫 tool_name，类型必须是字符串（str）。= Field(description="...")：给这个字段添加额外的元数据，比如描述信息、默认值、校验规则等。
    tool_name: str = Field(description="LangChain开发工具的名称，如 LangSmith")
    function: str = Field(description="工具的核心功能，30字以内")
    difficulty: str = Field(description="学习难度，仅可选：简单 / 中等 / 复杂")

# 3. 创建解析器
parser = PydanticOutputParser(pydantic_object=ToolInfo)

# 4. Prompt + Chain
prompt = PromptTemplate(
    template="{user_input}，严格按照要求输出。\n{format_instructions}",
    input_variables=["user_input"],
    partial_variables={
        "format_instructions": parser.get_format_instructions()
    }
)

chain = prompt | llm | parser
result = chain.invoke({"user_input": "请介绍1个 Python 开发工具"})

print("解析后的结构化数据（Pydantic 模型对象）：")
print(result)

print("字段校验 difficulty：", result.difficulty)

print("转化为字典：", result.model_dump())
