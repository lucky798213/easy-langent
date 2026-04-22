# 模拟面试智能体 —— 完整实现提示词

> 将此文件完整发送给 AI，它将根据以下规格实现整个面试流程。
> 前置条件：PDF 加载、分块、FAISS 向量库构建已完成，向量库对象名为 `vector_db`。

---

## 一、项目背景与已完成部分

我已经完成了以下模块，**不需要重新实现**：

```python
# 已完成：加载 PDF、按 section 分块、向量化存入 FAISS
# vector_db 是一个 FAISS 对象，每个 chunk 带有 metadata: {"section": "..."}
# 可用的 section 名称：work_experience / internship / projects / skills / education
```

**现在需要你实现**：从向量库检索开始，到面试问答循环、记忆管理、Tool 调用、报告生成的完整流程。

---

## 二、技术选型与组件清单

请严格使用以下 LangChain 组件，**不要替换为其他封装**：

| 用途 | 组件 | 导入路径 |
|---|---|---|
| 主 LLM | `ChatOpenAI` | `from langchain_openai import ChatOpenAI` |
| Prompt 模板 | `ChatPromptTemplate` | `from langchain_core.prompts import ChatPromptTemplate` |
| 历史消息占位 | `MessagesPlaceholder` | `from langchain_core.prompts import MessagesPlaceholder` |
| 模块对话记忆 | `ConversationBufferMemory` | `from langchain.memory import ConversationBufferMemory` |
| Tool 定义 | `@tool` 装饰器 | `from langchain_core.tools import tool` |
| Tool 绑定 | `llm.bind_tools()` | ChatOpenAI 原生方法 |
| 消息类型 | `HumanMessage` `AIMessage` `SystemMessage` | `from langchain_core.messages import ...` |
| MMR 检索 | `vector_db.max_marginal_relevance_search()` | FAISS 实例方法 |

**不要使用**：`ConversationalRetrievalChain`、`AgentExecutor`、`LLMChain`、`RetrievalQA`。这些封装太重，无法精确控制本项目的模块切换和记忆清空逻辑。

---

## 三、全局数据结构

```python
# 运行时全局状态（不需要持久化，面试结束后写文件）
module_summaries: list[dict] = []   # 每个模块结束后追加结构化摘要
weak_log: list[dict] = []           # 每次 record_weak_answer 调用后追加

# 模块配置（按此顺序执行，动态跳过不存在的 section）
MODULE_CONFIG = [
    {"id": "intro",      "name": "开场自我介绍",   "sections": [],                          "max_q": 1},
    {"id": "work",       "name": "工作与实习经历",  "sections": ["work_experience","internship"], "max_q": 3},
    {"id": "projects",   "name": "项目经历",        "sections": ["projects"],                "max_q": 3},
    {"id": "skills",     "name": "技术技能",        "sections": ["skills"],                  "max_q": 2},
    {"id": "cross",      "name": "综合追问",        "sections": [],                          "max_q": 2},
]
```

---

## 四、检索层实现

```python
def get_module_context(vector_db, sections: list[str], query: str, k: int = 4) -> str:
    """
    按 section 过滤 + MMR 检索，返回拼接好的简历片段字符串。
    sections 为空（开场/综合追问）时返回空字符串。
    
    关键参数说明：
    - filter={"section": section}  ← FAISS 后过滤，只返回该 section 的 chunk
    - lambda_mult=0.4              ← 偏多样性，避免返回重复 chunk
    - fetch_k=50                   ← 先取 50 个候选，再从中 MMR 筛 k 个
      （fetch_k 必须足够大，否则 filter 后可能返回空）
    """
    if not sections:
        return ""
    
    results = []
    for section in sections:
        hits = vector_db.max_marginal_relevance_search(
            query,
            k=k,
            fetch_k=50,
            lambda_mult=0.4,
            filter={"section": section}
        )
        results.extend(hits)
    
    return "\n\n---\n\n".join(doc.page_content for doc in results)
```

---

## 五、Tool 定义

**实现两个 Tool，注意 docstring 必须清晰，LLM 依靠 docstring 判断何时调用：**

```python
from langchain_core.tools import tool
import json

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
```

---

## 六、LLM 初始化与 Tool 绑定

```python
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4o", temperature=0.7)

# 绑定工具后，LLM 在合适时机会自动输出 tool_calls 而非普通文本
llm_with_tools = llm.bind_tools([record_weak_answer, save_module_summary])
```

---

## 七、Prompt 模板构建

**使用以下函数动态构建每轮对话的 prompt：**

```python
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

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
```

---

## 八、项目内置提示词（全部预设好，直接使用）

### 8.1 面试官 System Prompt 模板

```python
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
```

### 8.2 各模块触发消息（替代用户第一条输入）

```python
# 每个模块开始时，用此 trigger 替代第一轮 user_input，让 AI 主动开口
# 用户不会看到这条消息，只看到 AI 的回复

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
```

### 8.3 最终报告生成 Prompt

```python
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
```

---

## 九、单模块问答循环实现

```python
from langchain.memory import ConversationBufferMemory

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
```

---

## 十、主流程入口

```python
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
    from langchain_core.messages import HumanMessage

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


def main(vector_db):
    """
    主入口。vector_db 由外部传入（已完成的 FAISS 向量库）。
    """
    # 动态过滤：跳过简历中不存在的 section 对应模块
    # 先查询 vector_db 中实际存在哪些 section
    all_docs = vector_db.similarity_search("", k=999)  # 取所有文档
    existing_sections = {doc.metadata.get("section") for doc in all_docs}

    active_modules = [
        m for m in MODULE_CONFIG
        if not m["sections"]  # 开场/综合追问不依赖 section，始终保留
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


if __name__ == "__main__":
    # 假设 vector_db 已由前置流程构建完毕
    # from your_existing_module import vector_db
    main(vector_db)
```

---

## 十一、数据流总结（组件串联顺序）

```
vector_db（已有）
    │
    ▼
get_module_context()
    FAISS.max_marginal_relevance_search(filter=section, lambda_mult=0.4)
    │ 返回：简历片段字符串
    ▼
build_prompt()
    ChatPromptTemplate.from_messages([system, MessagesPlaceholder, human])
    │ 注入：module_context + prior_summaries + memory.chat_memory.messages
    ▼
prompt.format_messages(history=..., user_input=...)
    │ 返回：完整 messages 列表
    ▼
llm_with_tools.invoke(messages)
    ChatOpenAI(gpt-4o).bind_tools([record_weak_answer, save_module_summary])
    │
    ├─ response.content    → print 给用户 + memory.chat_memory.add_ai_message()
    │                          ConversationBufferMemory（模块级，结束即丢）
    │
    └─ response.tool_calls
           ├─ record_weak_answer  → weak_log.append()
           └─ save_module_summary → module_summaries.append() + return（结束模块）
    │
    ▼（所有模块完成后）
generate_final_report()
    REPORT_PROMPT_TEMPLATE.format(module_summaries, weak_log)
    llm.invoke([HumanMessage(prompt)])   ← 无 tools，无 memory
    │
    ▼
interview_report.md + weak_answers.json
```

---

## 十二、依赖安装

```bash
pip install langchain langchain-openai langchain-community \
            langchain-huggingface faiss-cpu pdfplumber pydantic openai
```

## 十三、环境变量

```bash
# .env 文件
OPENAI_API_KEY=sk-...
```

```python
from dotenv import load_dotenv
load_dotenv()
```