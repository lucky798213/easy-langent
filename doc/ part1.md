# LangChain 与 LangGraph 学习笔记

## 两者区别

| 特性 | LangChain | LangGraph |
|------|-----------|-----------|
| 适用场景 | 串行流程 | 复杂流程（图结构） |
| 核心能力 | 链式调用 | 循环、分支、状态管理 |
| 定位 | 基础编排 | 真正的 Agent 框架 |

---

## LangChain

```python
llm = ChatOpenAI(
    api_key=API_KEY,
    base_url=BASE_URL,
    model="deepseek-chat",  # 注意：根据你使用的模型修改名称！！！！ 后面章节不再继续说明
    temperature=0.3
)
```

实例化一个 `ChatOpenAI` 对象

```python
response = llm.invoke("你好")  # 调用模型
print(response.content)  # 打印模型回复的内容
print(response)
```

---

## LangGraph

```python
workflow = StateGraph(WorkflowState)
```

构建这个工作流并定义了他的输入

```python
workflow.add_node("generate", generate_advice)
```

添加这个工作流中的节点，第一个变量是代表这个节点的名称，第二个变量是该节点具体要执行的程序

```python
workflow.add_edge(START, "generate")
workflow.add_edge("generate", "simplify")
workflow.add_edge("simplify", END)
```

这里是将各个节点进行连线，进而构成图

```python
app = workflow.compile()
```

将这个工作流转化为可执行的应用（app）

```python
result = app.invoke({"user_role": "高校学生"})
```

执行工作流，`result` 存着整个工作流执行完后的最终 state

---

## Python 基础知识补丁

- 虚拟环境中有系统包和自己安装的包是吧，然后开发的程序中读取的包都是来自虚拟环境中的包

- `import` 是导入一个模块，模块中包含了多个函数、类、变量等

- `from ... import ...` 表示从...模块中导入...函数、类、变量等

- `class WorkflowState(TypedDict, total=False)` 这里是表示创建一个类，类中包含了多个属性，每个属性都有一个类型，这里 `TypedDict` 是一个字典，`total=False` 表示允许有未定义的属性
