rag模拟面试系统
解析并切分PDF文件
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os

# 1. 加载PDF，得到每页的Document列表
pdf_path = os.path.join("knowledge_base", "test.pdf")
loader = PyPDFLoader(pdf_path)
pdf_docs = loader.load_and_split()   # 每个元素是一页的Document

# 2. 初始化文本分割器
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,        # 每个块的最大字符数（可调整）
    chunk_overlap=50,      # 块之间的重叠字符数，保持语义连贯
    separators=["\n\n", "\n", "。", "！", "？", "，", "；", "、"],  # 默认分隔符，优先按段落分割
    length_function=len,   # 计算文本长度的方法
)

# 3. 对每个Document进行切割，得到所有chunks
chunks = text_splitter.split_documents(pdf_docs)

# 查看切割结果
print(f"原始页数：{len(pdf_docs)}")
print(f"切割后块数：{len(chunks)}")
print(f"第一个块的内容预览：{chunks[0].page_content[:200]}...")
print(f"第一个块的元数据：{chunks[0].metadata}")  # 会继承原页面的元数据，并可能添加分割信息

向量化
import os
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings

txt_path = os.path.join("knowledge_base", "test.txt")
if not os.path.exists(txt_path):
    raise FileNotFoundError(f"文档文件不存在：{txt_path}")

# 加载文本文档
loader = TextLoader(txt_path, encoding="utf-8")
txt_docs: list[Document] = loader.load()

# 文本分割
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,          
    chunk_overlap=50,        
    length_function=len,     
    is_separator_regex=False 
)
split_docs: list[Document] = text_splitter.split_documents(txt_docs)
print(f"分割后的文本片段数：{len(split_docs)}")

# 3. ✅【唯一修改处】用你本地真实的模型路径！
embedding_model_name = "/Users/liangzhancheng/.cache/modelscope/hub/models/Qwen/Qwen3-Embedding-0.6B"

embeddings = HuggingFaceEmbeddings(
    model_name=embedding_model_name,
    model_kwargs={
        "device": "cpu" 
    },
    encode_kwargs={
        "normalize_embeddings": True
    }
)

# 4. 构建FAISS向量库
try:
    vector_db = FAISS.from_documents(
        documents=split_docs,
        embedding=embeddings,
    )
    vector_db.save_local(
        folder_path="./faiss_db",
        index_name="local_cpu_faiss_index"
    )
    print("向量存储完成！数据已保存到 ./faiss_db 文件夹")
except Exception as e:
    raise RuntimeError(f"构建/保存向量库失败：{str(e)}")

# 5. 检索测试
query = "LangChain的链式工作流有哪些类型？"
try:
    retrieved_docs_with_scores = vector_db.similarity_search_with_score(query, k=3)
    
    print(f"\n与问题「{query}」最相关的3个文本片段：")
    for i, (doc, score) in enumerate(retrieved_docs_with_scores):
        print(f"\n片段{i+1}：")
        print(f"内容：{doc.page_content}")
        print(f"相关性评分：{round(score, 4)}")
        print(f"来源：{doc.metadata.get('source', '未知')}")
except Exception as e:
    raise RuntimeError(f"检索失败：{str(e)}")

流程设计
假定这个文件就是固定的位置。首先我们就是先加载这个文件，然后对这个文件进行切分，最后对切分后的文本进行向量化。面试程序启动之后，按顺序执行，如果有实习经历或工作经历，就内置自己先发起实习经历和工作经历的向量请求，返回得到实习经历和工作经历的内容后，ai就根据这些内容提出相关问题，等待用户的回答。在接收到用户回答后，生成反馈，再提出下一个问题，每个模块最多不要超过5个问题。接下就到项目经历和个人技能模块。和前面执行相同的步骤。每到下一个模块之后前一个模块的记忆清空，在清空前先对面试这个模块的内容进行总结，生成总结文本存储起来再清空记忆。最终将文本返回给用户