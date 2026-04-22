from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableMap, RunnablePassthrough
from dotenv import load_dotenv
from operator import itemgetter   
import os

# 1. 初始化模型
load_dotenv()
llm = ChatOpenAI(
    api_key=os.getenv("API_KEY"),
    base_url="https://api.deepseek.com",
    model="deepseek-chat",
    temperature=0.3
)


#根据分类标签列表，对新闻文本进行分类，输出分类结果（category）
new_category_prompt = PromptTemplate(
    input_variables=["news_text", "category_list"],
    template="理解这段新闻{news_text}，看看他属于哪种类型的新闻，{category_list}，只返回新闻类型，不要返回其他内容"
)
new_category_chain = new_category_prompt | llm 


#根据分类结果和新闻文本，提取该类新闻的核心事件
core_message_prompt = PromptTemplate(
    input_variables=["news_text", "category"],
    template="根据新闻类型{new_category}，提取该类新闻的核心事件（如科技新闻提取“技术突破、产品发布”等，财经新闻提取“政策变化、企业动态”等）：{news_text}"
)
core_message_chain = core_message_prompt | llm | (lambda x: x.content)

# 3. 多输入多输出线性链（教学标准版）
overall_chain = (
    # Step 1：生成分类 + 透传原始输入
    RunnableMap({
        "news_text": itemgetter("news_text"),
        "new_category": new_category_chain | (lambda x: x.content),
    })
    # Step 2：提取核心事件
    | core_message_chain
)

# 4. 执行
input_data = {
    "news_text": "联合国气候变化大会达成新协议，190 个国家承诺 2030 年前减少碳排放 50%。",
    "category_list": ["科技", "财经", "娱乐", "体育", "政治", "社会", "国际", "健康"]
}
# 预期分类：国际
# 预期核心事件：190 个国家承诺 2030 年前减少碳排放 50%

result = overall_chain.invoke(input_data)

print("提取出的核心事件：")
print(result)