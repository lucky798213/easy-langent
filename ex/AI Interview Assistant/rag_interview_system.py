"""
RAG 模拟面试系统
整合 PDF 解析、文本切分、向量化和检索功能
"""

import os
import json
from pathlib import Path
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.memory import ConversationBufferMemory
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage
import re

SECTION_KEYWORDS = {
    "work_experience": ["工作经历", "工作经验", "Work Experience"],
    "internship":      ["实习经历", "实习", "Internship"],
    "projects":        ["项目经历", "项目", "Projects"],
    "skills":          ["技术栈", "专业技能", "技能", "Skills"],
    "education":       ["教育背景", "Education"],
}

# 运行时全局状态（不需要持久化，面试结束后写文件）
module_summaries: list[dict] = []#用来记录每个模块面试后的得分
weak_log: list[dict] = []

# 模块配置（按此顺序执行，动态跳过不存在的 section）
MODULE_CONFIG = [
    {"id": "intro",      "name": "开场自我介绍",   "sections": [],                          "max_q": 1},
    {"id": "work",       "name": "工作与实习经历",  "sections": ["work_experience","internship"], "max_q": 3},
    {"id": "projects",   "name": "项目经历",        "sections": ["projects"],                "max_q": 3},
    {"id": "skills",     "name": "技术技能",        "sections": ["skills"],                  "max_q": 2},
    {"id": "cross",      "name": "综合追问",        "sections": [],                          "max_q": 2},
]

SYSTEM_PROMPT_TEMPLATE = """
你是一位经验丰富、专业友好的技术面试官。你正在对候选人进行结构化技术面试。

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
【当前模块简历内容】
{module_context}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
【前序模块摘要（用于跨模块关联）】
{prior_summaries}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
【当前面试状态】
- 当前模块：{module_name}（模块ID：{module_id}）
- 本模块已提问：{question_count} 题 / 上限 {max_q} 题

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
【行为规则 - 严格遵守】

提问规则：
- 每次只提出一个问题，等待用户回答后再继续
- 问题必须基于简历内容，针对性强，不问泛泛而谈的问题
- 问题由浅入深，先确认事实，再追问细节和思考

回答评估规则：
- 回答完整清晰 → 给予简短正向反馈（1句），直接进入下一问
- 回答模糊/缺细节 → 先追问一次（"能具体说说...吗？"）
- 回答明显不足 → 给出建议后，立即调用 record_weak_answer 工具记录
- 不要过度追问同一个问题，最多追问 1 次

模块结束规则：
- 当已提问数量达到 {max_q} 题上限时，必须调用 save_module_summary 工具
- 调用前先说："好的，{module_name}部分我们就聊到这里。"
- save_module_summary 的 summary_json 参数必须是合法 JSON 字符串
- 调用完成后告知用户："接下来我们进入下一个环节。"

语气风格：
- 专业但不冷漠，适当使用"嗯""好的"等自然过渡语
- 不要说"作为AI"或暴露自己是 AI
- 不要照本宣科地读简历，要像真实面试官一样交流
"""

MODULE_TRIGGERS = {
    "intro": (
        "请用自然的方式开场，先做简短的自我介绍（作为面试官），"
        "然后邀请候选人进行自我介绍。语气轻松，不超过2句话。"
    ),
    "work": (
        "请根据候选人的工作/实习经历，直接提出第一个有针对性的问题。"
        "不要说'让我们开始'之类的废话，直接提问。"
    ),
    "projects": (
        "我们现在来聊聊你的项目经历。请根据简历中的项目内容，"
        "选择最有技术含量的一个项目，直接提出第一个问题。"
    ),
    "skills": (
        "接下来我们来看看你的技术技能。请根据简历中的技术栈，"
        "选择一个核心技术点直接发起提问。"
    ),
    "cross": (
        "在前面的交流中我对你有了初步了解。现在我想问一个综合性的问题，"
        "将你的不同经历关联起来。请根据所有前序模块摘要，提出1-2个跨模块的深度问题。"
    ),
}

REPORT_PROMPT_TEMPLATE = """
你是一位资深技术面试官，请根据以下完整面试数据，生成一份专业的候选人评估报告。

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
【各模块面试摘要】
{summaries_json}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
【弱项记录（回答质量较差的题目）】
{weak_json}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
【报告输出要求】

请严格按以下 Markdown 格式输出，不要添加额外章节：

---

# 面试评估报告

## 总体评分：X/10
> 一句话总体评语（20字以内，客观中肯）

---

## 各模块表现

### 工作/实习经历（X/10）
**亮点：** ...  
**不足：** ...  
**建议：** ...

### 项目经历（X/10）
**亮点：** ...  
**不足：** ...  
**建议：** ...

### 技术技能（X/10）
**亮点：** ...  
**不足：** ...  
**建议：** ...

---

## 核心技术能力评估
（列出候选人提到的主要技术，每项给出简短的能力评价）

---

## 需要重点改进的 3 个方向
1. **方向一**：具体建议...
2. **方向二**：具体建议...
3. **方向三**：具体建议...

---

## 面试整体建议
（2-3句话，给候选人最重要的一条面试表现建议）

---

语气要求：专业、建设性、有温度，不要过于严苛，也不要空洞夸奖。
"""


# 1. 加载并解析简历，按 section 切分，返回切分好的 Document 列表
def load_and_parse_resume(pdf_path: str) -> list[Document]:
    # 1. PyPDFLoader 加载，拿全文
    loader = PyPDFLoader(pdf_path)
    pages = loader.load()                          # 保留原始 loader，不变
    full_text = "\n".join(p.page_content for p in pages)
    print(full_text)

    # 2. 按 section 关键词切分
    sections: dict[str, list[str]] = {}
    current_key = "other"
    current_lines: list[str] = []

    #一行行遍历文本，然后再检查这一行中有没有项目等关键字（SECTION_KEYWORDS.items）
    for line in full_text.splitlines():
        matched = False
        for key, keywords in SECTION_KEYWORDS.items():
            if any(kw in line for kw in keywords):#只要有一个关键字对上就返回
                if current_lines:#检查 current_lines 是否非空（Python 中空列表等于 False）
                    sections.setdefault(current_key, []).extend(current_lines)
                current_key = key
                current_lines = []
                matched = True
                break
        if not matched:
            current_lines.append(line)

    if current_lines:
        sections.setdefault(current_key, []).extend(current_lines)

    # 3. 转成带 metadata 的 Document
    docs: list[Document] = []
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=50,
        separators=["\n\n", "\n", "。", "！", "？", "，", "；"],
    )

    for section_name, lines in sections.items():
        content = "\n".join(lines).strip()
        if not content:
            continue

        if section_name == "projects":
            # 项目按编号/名称切，不按字数
            sub_chunks = re.split(r"\n(?=\d+[.、]|[A-Z][A-Za-z ]{3,}:)", content)
            for i, chunk in enumerate(sub_chunks):
                if chunk.strip():
                    docs.append(Document(
                        page_content=chunk.strip(),
                        metadata={"section": section_name, "project_index": i, "source": pdf_path}
                    ))
        elif len(content) > 600:
            # 长 section 才用 splitter 切
            for chunk in splitter.split_text(content):
                docs.append(Document(
                    page_content=chunk,
                    metadata={"section": section_name, "source": pdf_path}
                ))
        else:
            # 短 section 整块保留
            docs.append(Document(
                page_content=content,
                metadata={"section": section_name, "source": pdf_path}
            ))

    print(f"解析完成，共 {len(docs)} 个 chunk，section 分布：")
    from collections import Counter
    counter = Counter(d.metadata["section"] for d in docs)
    for sec, count in counter.items():
        print(f"  {sec}: {count} 个 chunk")

    # Debug: 打印每个 chunk 的内容预览
    print(f"\n{'='*60}")
    print("切分后的文本内容预览：")
    print(f"{'='*60}")
    for i, doc in enumerate(docs):
        section = doc.metadata.get("section", "unknown")
        preview = doc.page_content[:200].replace("\n", " ")
        print(f"\n[Chunk {i+1}] Section: {section}")
        print(f"内容: {preview}{'...' if len(doc.page_content) > 200 else ''}")
    print(f"\n{'='*60}")

    return docs

def create_vector_store(chunks: list[Document], 
                        embedding_model_path: str,
                        save_path: str = "./faiss_db",
                        index_name: str = "interview_index") -> FAISS:
    """
    创建向量数据库
    
    Args:
        chunks: 文本块列表
        embedding_model_path: 本地嵌入模型路径
        save_path: 向量库保存路径
        index_name: 索引名称
        
    Returns:
        FAISS 向量数据库对象
    """
    print(f"\n正在初始化嵌入模型: {embedding_model_path}")
    
    embeddings = HuggingFaceEmbeddings(
        model_name=embedding_model_path,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )
    
    print("正在构建 FAISS 向量库...")
    vector_db = FAISS.from_documents(
        documents=chunks,
        embedding=embeddings,
    )
    
    # 保存向量库
    vector_db.save_local(
        folder_path=save_path,
        index_name=index_name
    )
    print(f"向量库存储完成！已保存到 {save_path} 文件夹")
    
    return vector_db

def get_module_context(vector_db: FAISS,
                       sections: list[str],
                       query: str,
                       k: int = 4) -> str:
    if not sections:
        return ""

    results: list[Document] = []
    for section in sections:
        # FAISS 的 MMR 检索 + metadata 过滤
        hits = vector_db.max_marginal_relevance_search(
            query,
            k=k,
            fetch_k=50,
            lambda_mult=0.4,
            filter={"section": section}   # 只在指定 section 里查
        )
        results.extend(hits)

    return "\n\n---\n\n".join(doc.page_content for doc in results)

def search_similar_documents(vector_db: FAISS, 
                             query: str, 
                             k: int = 3) -> list[tuple[Document, float]]:
    """
    检索与查询最相似的文档
    
    Args:
        vector_db: FAISS 向量数据库
        query: 查询问题
        k: 返回的最相似文档数量
        
    Returns:
        (文档, 相似度分数) 元组列表
    """
    print(f"\n检索问题: 「{query}」")
    results = vector_db.similarity_search_with_score(query, k=k)
    
    print(f"找到 {len(results)} 个最相关的文本片段：")
    for i, (doc, score) in enumerate(results):
        print(f"\n{'='*50}")
        print(f"片段 {i+1} (相似度: {round(score, 4)}):")
        print(f"{'='*50}")
        print(f"内容: {doc.page_content}")
        print(f"来源: {doc.metadata.get('source', '未知')}")
    
    return results


@tool
def record_weak_answer(question: str, user_answer: str, suggestion: str) -> str:
    """
    当评估用户对某道面试题的回答质量较差时调用此工具。
    判断标准：回答内容模糊、缺乏细节、逻辑不清、或明显偏离问题。
    
    Args:
        question: 面试官提出的原始问题
        user_answer: 用户的原始回答内容  
        suggestion: 针对此问题的改进建议和参考答案方向
    """
    entry = {"question": question, "user_answer": user_answer, "suggestion": suggestion}
    weak_log.append(entry)
    return f"已记录弱项问题：{question[:40]}..."


@tool  
def save_module_summary(summary_json: str) -> str:
    """
    在当前面试模块的问题数量达到上限，或该模块内容已充分考察后调用。
    调用此工具意味着当前模块面试结束，系统将清空本模块对话记忆并进入下一模块。
    
    Args:
        summary_json: 严格按以下 JSON 格式输出的模块总结字符串：
        {
            "module": "模块id（如 work / projects / skills）",
            "key_tech": ["提到的技术1", "技术2"],
            "strengths": ["表现好的点1", "点2"],
            "weak_points": ["薄弱点1", "点2"],
            "score": 7
        }
    """
    try:
        data = json.loads(summary_json)
        module_summaries.append(data)
        return f"模块 [{data.get('module')}] 摘要已保存，得分 {data.get('score')}/10"
    except Exception as e:
        return f"摘要保存失败：{e}"


llm = ChatOpenAI(
    api_key=API_KEY,
    base_url=BASE_URL,
    model="deepseek-chat",  # 注意：根据你使用的模型修改名称！！！！ 后面章节不再继续说明
    temperature=0.3
)

# 绑定工具后，LLM 在合适时机会自动输出 tool_calls 而非普通文本
llm_with_tools = llm.bind_tools([record_weak_answer, save_module_summary])


def build_prompt(module_context: str, prior_summaries: str,
                 module_name: str, module_id: str,
                 question_count: int, max_q: int) -> ChatPromptTemplate:
    
    system_text = SYSTEM_PROMPT_TEMPLATE.format(
        module_context=module_context or "（本模块无需检索简历内容）",
        prior_summaries=prior_summaries or "（暂无前序模块摘要）",
        module_name=module_name,
        module_id=module_id,
        question_count=question_count,
        max_q=max_q,
    )
    
    return ChatPromptTemplate.from_messages([
        ("system", system_text),
        MessagesPlaceholder(variable_name="history"),  # ← 注入 memory 中的历史消息
        ("human", "{user_input}"),
    ])


def run_module(module: dict, vector_db, prior_summaries_text: str) -> None:
    """
    执行单个面试模块的完整问答循环。
    memory 在函数内部创建，函数返回后自动销毁（模块记忆自动清空）。
    """
    # ① 每个模块独立创建 memory，函数结束后自动丢弃
    memory = ConversationBufferMemory(return_messages=True)
    question_count = 0

    # ② 检索当前模块的简历内容
    context = get_module_context(
        vector_db=vector_db,
        sections=module["sections"],
        query=module["name"],
    )

    print(f"\n{'━'*50}")
    print(f"  模块：{module['name']}")
    print(f"{'━'*50}\n")

    # ③ 第一轮用 trigger 消息，让 AI 主动开口（用户看不到此消息）
    user_input = MODULE_TRIGGERS[module["id"]]

    while True:
        # ④ 动态构建 prompt（每轮都重新构建，question_count 实时更新）
        prompt = build_prompt(
            module_context=context,
            prior_summaries=prior_summaries_text,
            module_name=module["name"],
            module_id=module["id"],
            question_count=question_count,
            max_q=module["max_q"],
        )

        # ⑤ 将 memory 历史 + 当前输入填入 prompt
        messages = prompt.format_messages(
            history=memory.chat_memory.messages,
            user_input=user_input,
        )

        # ⑥ 调用绑定了 Tool 的 LLM
        response = llm_with_tools.invoke(messages)

        # ⑦ 处理 Tool 调用
        if response.tool_calls:
            for tc in response.tool_calls:
                if tc["name"] == "record_weak_answer":
                    result = record_weak_answer.invoke(tc["args"])
                    print(f"  [记录弱项] {result}")

                elif tc["name"] == "save_module_summary":
                    result = save_module_summary.invoke(tc["args"])
                    print(f"  [模块摘要] {result}")
                    # save_module_summary 被调用 = 模块结束信号，立即退出循环
                    return

        # ⑧ 输出 AI 的文本回复
        ai_text = response.content
        if ai_text:
            print(f"面试官：{ai_text}\n")
            memory.chat_memory.add_ai_message(ai_text)

        # ⑨ 兜底：防止 LLM 忘记调工具时死循环
        if question_count >= module["max_q"]:
            print("  [系统] 已达问题上限，强制结束本模块\n")
            # 强制生成摘要
            force_summary = json.dumps({
                "module": module["id"],
                "key_tech": [], "strengths": [], "weak_points": [],
                "score": 5
            }, ensure_ascii=False)
            save_module_summary.invoke({"summary_json": force_summary})
            return

        # ⑩ 等待用户真实输入（第一轮之后才真正等待）
        user_input = input("你：").strip()
        if not user_input:
            continue

        memory.chat_memory.add_user_message(user_input)
        question_count += 1


def build_prior_summaries_text() -> str:
    """将已完成模块的摘要格式化为文本，注入下一模块的 system prompt"""
    if not module_summaries:
        return ""
    lines = []
    for s in module_summaries:
        lines.append(
            f"[{s['module']}] 得分:{s.get('score','-')}/10 | "
            f"技术:{s.get('key_tech',[])} | "
            f"弱点:{s.get('weak_points',[])}"
        )
    return "\n".join(lines)


def generate_final_report() -> str:
    """汇总所有摘要和弱项，生成最终 Markdown 报告"""
    prompt = REPORT_PROMPT_TEMPLATE.format(
        summaries_json=json.dumps(module_summaries, ensure_ascii=False, indent=2),
        weak_json=json.dumps(weak_log, ensure_ascii=False, indent=2),
    )

    # 报告生成不需要记忆也不需要工具，直接用基础 llm
    response = llm.invoke([HumanMessage(content=prompt)])
    report_md = response.content

    Path("interview_report.md").write_text(report_md, encoding="utf-8")
    Path("weak_answers.json").write_text(
        json.dumps(weak_log, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return report_md


def main_interview(vector_db):
    """
    主入口。vector_db 由外部传入（已完成的 FAISS 向量库）。
    """
    # 动态过滤：跳过简历中不存在的 section 对应模块
    # 先查询 vector_db 中实际存在哪些 section
    all_docs = vector_db.similarity_search("", k=999)  # 取所有文档
    existing_sections = {doc.metadata.get("section") for doc in all_docs}

    active_modules = [
        m for m in MODULE_CONFIG
        if not m["sections"]  # 如果这个模块不需要 section → 直接通过，否则 → 看它依赖的 section 在不在简历里
        or any(s in existing_sections for s in m["sections"])
    ]

    print(f"检测到简历包含的模块：{existing_sections}")
    print(f"本次面试将执行 {len(active_modules)} 个模块\n")

    # 逐模块执行
    for module in active_modules:
        prior_text = build_prior_summaries_text()
        run_module(module=module, vector_db=vector_db, prior_summaries_text=prior_text)

    # 生成最终报告
    print("\n" + "━"*50)
    print("  面试结束，正在生成评估报告...")
    print("━"*50 + "\n")
    report = generate_final_report()
    print(report)
    print("\n报告已保存至 interview_report.md")
    print("弱项问卷已保存至 weak_answers.json")


def main():
    """主函数：完整的 RAG 面试系统流程"""
    
    # 加载环境变量
    load_dotenv()
    
    # ==================== 配置区域 ====================
    # PDF 文件路径
    pdf_path = "test.pdf"
    
    # 嵌入模型路径（请修改为你本地的真实路径）
    embedding_model_path = "/Users/liangzhancheng/.cache/modelscope/hub/models/Qwen/Qwen3-Embedding-0.6B"
    
    # 向量库保存路径
    vector_db_path = "./faiss_db"
    index_name = "interview_index"
    
    # 测试查询问题
    test_query = "项目"
    # =================================================
    
    print("="*60)
    print("RAG 模拟面试系统启动")
    print("="*60)
    
    # 步骤 1: 检查 PDF 文件是否存在
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF 文件不存在：{pdf_path}")
    
    # 步骤 2: 加载并切分 PDF
    print("\n【步骤 1/4】加载 PDF 文档并切分文本")
    pdf_docs = load_and_parse_resume(pdf_path)
    
    # 步骤 4: 创建向量库
    print("\n【步骤 2/4】创建向量数据库...")
    vector_db = create_vector_store(
        chunks=pdf_docs,
        embedding_model_path=embedding_model_path,
        save_path=vector_db_path,
        index_name=index_name
    )
    
    # 步骤 5: 检索测试
    print("\n【步骤 3/4】检索测试...")
    search_similar_documents(vector_db, test_query, k=2)
    
    print("\n【步骤 4/4】开始面试...")
    main_interview(vector_db)
    
    print("\n" + "="*60)
    print("RAG 系统运行完成！")
    print(f"向量库已保存至: {vector_db_path}")
    print("="*60)


if __name__ == "__main__":
    main()
