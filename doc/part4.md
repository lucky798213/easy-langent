langchain链路工作流
主要通过｜来连接不同组件形成一个工作流，组件的底层类是runnable，主要就是构建提示词，+llm。要分分支就用RunnableMap。在RunnableMap之后的组件，能接收到的内容，就只有RunnableMap中定义的键值对。
langchain错误处理机制
重试机制
retry_chain = base_chain.with_retry(
    stop_after_attempt=3,          # 最多重试 3 次
    wait_exponential_jitter=True,  # 指数退避 + 抖动（推荐）
    retry_if_exception_type=(
        ConnectionError,
        TimeoutError,
    ),
)直接在已实现的链路调用with_retry，
异常捕获（解决可预知的错误），直接通过python的try except捕获即可
降级：构建两条链路，一条核心链，一条降解链。
chain_with_fallback: RunnableWithFallbacks = core_chain.with_fallbacks(
    fallbacks=[fallback_chain],
    exceptions_to_handle=(ConnectionError, TimeoutError),# ✅ 官方推荐：只捕获临时错误或网络错误
)核心链通过调用这个with_fallbacks方法就可以实现降级，fallbacks备有链路，核心链路失败时会注意调用降级链路，exceptions_to_handle只要核心链路抛出这些异常时，才会调用降级链路
rag构建
from langchain_text_splitters import RecursiveCharacterTextSplitter 使用这个
# 2. 初始化分割器（LangChain推荐参数）
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,          # 中文片段推荐长度：200-500字
    chunk_overlap=50,        # 重叠长度：建议为chunk_size的10%-20%，避免跨片段语义丢失
    length_function=len,     # 中文用len计数，英文可改用tiktoken.count_tokens
    separators=["\n\n", "\n", "。", "！", "？", "，", "；", "、"]  # 中文推荐分隔符优先级
)

# 3. 执行分割（split_documents为官方推荐方法，接收Document列表）
split_docs = text_splitter.split_documents(txt_docs)
