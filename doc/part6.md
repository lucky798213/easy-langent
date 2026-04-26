# LangGraph 学习笔记

## 一、Python 类型注解基础

### 1.1 TypedDict 与继承

```python
from typing import TypedDict

class TaskState(TypedDict):
    """括号表示继承 TypedDict"""
    ...
```

> 💡 **注意**：节点函数中的 `state: TaskState` 只是 Python 类型注解，**运行时没有任何效果**。

### 1.2 Python 类型系统特点

| 特性 | 说明 |
|------|------|
| **动态类型** | 类型在运行时确定 |
| **函数返回** | 不需要声明返回类型 |
| **类型注解** | 可选的，仅用于提示/工具 |
| **运行行为** | 不检查注解，注解被忽略 |

---

## 二、LangGraph 核心概念

### 2.1 状态（State）是什么？

LangGraph 中的状态可以理解为**一块共享内存**，图中的每一个节点都可以访问和修改这个状态。

### 2.2 定义状态的两种方式

#### 方式一：TypedDict（推荐）

```python
from typing import TypedDict, NotRequired

class TaskState(TypedDict):
    user_query: str                    # 用户原始查询（必需）
    tool_result: NotRequired[str]      # 工具调用结果（可选）
    final_answer: NotRequired[str]     # 最终回答（可选）
    progress: NotRequired[int]         # 任务进度百分比（可选）
```

#### 方式二：Pydantic BaseModel

```python
from pydantic import BaseModel, Field
from typing import Optional

class TaskState(BaseModel):
    user_query: str = Field(description="用户原始查询")
    tool_result: Optional[str] = Field(default=None, description="工具调用结果")
    final_answer: Optional[str] = Field(default=None, description="最终回答")
    progress: Optional[int] = Field(default=None, description="任务进度百分比")
```

### 2.3 创建图

```python
from langgraph.graph import StateGraph

# 创建图时传入状态类型
graph = StateGraph(TaskState)
```

---

## 三、节点（Node）开发规范

### 3.1 状态更新原则

> ⚠️ **重要**：节点函数**不能直接修改状态**，只能返回更新的部分。
>
> 每个节点只负责"声明我要改什么"，由图引擎统一合并。

```python
# ✅ 正确做法：返回要更新的字段
def my_node(state: TaskState) -> dict:
    return {"tool_result": "查询结果"}

# ❌ 错误做法：直接修改状态
def my_node(state: TaskState) -> None:
    state["tool_result"] = "查询结果"  # 不要这样做！
```

### 3.2 节点类型分类

| 类型 | 说明 | 示例 |
|------|------|------|
| **LLM 节点** | 调用语言模型生成回复 | 对话生成、意图识别 |
| **工具节点** | 调用外部工具/函数 | 搜索、计算、API 调用 |
| **数据处理节点** | 转换或处理数据 | 格式化、过滤、聚合 |

---

## 四、条件边（Conditional Edges）

条件边用于在节点之间添加路由逻辑，根据状态决定执行路径。

### 4.1 基本语法

```python
builder.add_conditional_edges(
    source,       # 必填：从哪个节点出发，字符串
    path,         # 必填：路由函数，接收 state，必须返回字符串
    path_map,     # 选填：字典，把路由函数返回值映射到节点名
)
```

### 4.2 使用方式

#### 方式一：不传 path_map（直接返回节点名）

```python
def decide(state):
    if state.get("score", 0) > 80:
        return "pass_node"
    else:
        return "retry_node"
    # ← 必须和 add_node 时的名字完全一致

builder.add_conditional_edges("judge", decide)
```

#### 方式二：传 path_map（返回值映射）

```python
def decide(state):
    if state.get("score", 0) > 80:
        return "high"   # 可以返回任意标识
    else:
        return "low"

builder.add_conditional_edges(
    "judge",
    decide,
    path_map={"high": "pass_node", "low": "retry_node"}  # 映射到实际节点
)
```

---

## 五、快速参考

```python
from typing import TypedDict, NotRequired
from langgraph.graph import StateGraph, END

# 1. 定义状态
class State(TypedDict):
    query: str
    result: NotRequired[str]

# 2. 定义节点
def process(state: State) -> dict:
    return {"result": f"处理: {state['query']}"}

# 3. 构建图
builder = StateGraph(State)
builder.add_node("process", process)
builder.set_entry_point("process")
builder.add_edge("process", END)

# 4. 编译运行
graph = builder.compile()
result = graph.invoke({"query": "Hello"})
```
