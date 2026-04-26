# LangGraph 进阶：多智能体架构与高级特性

## 一、Python 语法补充

### 1.1 Optional 类型

```python
from typing import Optional

# Optional[int] 表示：可以是 int，也可以是 None
value: Optional[int] = 42      # ✅ 合法
value: Optional[int] = None    # ✅ 合法
```

### 1.2 函数参数规则

| 要素 | 是否必须 | 说明 |
|------|----------|------|
| **参数名** | ✅ 必须 | 函数体内使用的占位符，如 `state`、`x`、`name` |
| **类型注解** | ❌ 可选 | 仅用于提示，Python 解释器不强制检查 |

```python
# 参数名必须有，类型注解可省略
def process(state):           # ✅ 合法（无类型注解）
    pass

def process(state: dict):     # ✅ 合法（有类型注解）
    pass
```

---

## 二、多智能体架构模式

### 2.1 中心化协作（Supervisor）

> 🎯 **核心逻辑**：有一个"主管智能体"（Supervisor）负责接收总任务、拆分任务、分配给不同的"员工智能体"，并汇总结果——就像公司里的"项目经理"，不干活，只协调。

**适用场景**：任务可明确拆分、需要统一协调的场景

**实现本质**：
- 一个主管节点通过路由函数判断当前状态
- 根据状态分配给不同的员工节点
- 员工节点执行完后回到主管节点
- 主管节点判断是否完成任务，决定退出或继续

```
┌─────────┐     ┌──────────┐     ┌─────────┐
│  START  │────▶│ Supervisor│◀────│  Worker │
└─────────┘     └────┬─────┘     └─────────┘
                     │
        ┌────────────┼────────────┐
        ▼            ▼            ▼
   ┌────────┐   ┌────────┐   ┌────────┐
   │Worker A│   │Worker B│   │Worker C│
   └───┬────┘   └───┬────┘   └───┬────┘
       └────────────┴────────────┘
                    │
                    ▼
              ┌─────────┐
              │   END   │
              └─────────┘
```

---

### 2.2 链式协作（Sequence）

> 🎯 **核心逻辑**：没有主管，多个智能体按"固定顺序"接力完成任务，每个智能体的输出作为下一个智能体的输入——就像"流水线生产"，上一道工序做完，交给下一道，直到完成。

**适用场景**：任务流程固定、顺序不可颠倒的场景

```
START ──▶ Node A ──▶ Node B ──▶ Node C ──▶ END
          │          │          │
          └──────────┴──────────┘
              顺序执行，不可跳跃
```

---

### 2.3 去中心化协作（Peer-to-peer）

> 🎯 **核心逻辑**：没有主管，每个智能体都是"平等的"，根据全局状态的变化，自主决定是否执行任务——就像"创业团队"，每个人都盯着项目目标，不用别人分配，自己主动干活。

**适用场景**：任务灵活、无法提前固定流程，需要智能体自主响应状态变化的场景

**实现本质**：
- 将每个智能体串成一条链
- 首尾相连，形成一个环
- 每次执行完一个后，通过路由函数判断是否结束
- 每个智能体根据全局状态自主决定是否执行任务

> ⚠️ **注意**：这还不是真正的去中心化，只是一种简化实现

```
        ┌─────────────────────────────────┐
        │                                 │
        ▼                                 │
   ┌─────────┐    ┌─────────┐    ┌────────┴─┐
   │ Agent A │───▶│ Agent B │───▶│ Agent C  │
   └─────────┘    └─────────┘    └──────────┘
        │                              │
        └──────────▶ END ◀─────────────┘
                  （条件退出）
```

---

## 三、子图（Subgraph）

子图的实现非常简单：先定义一个子图（StateGraph），编译后作为一个"节点"添加到主图中即可。

```python
from langgraph.graph import StateGraph, END

# 1. 定义子图
sub_builder = StateGraph(State)
sub_builder.add_node("step1", step1_func)
sub_builder.add_node("step2", step2_func)
sub_builder.set_entry_point("step1")
sub_builder.add_edge("step1", "step2")
sub_builder.add_edge("step2", END)

subgraph = sub_builder.compile()  # 编译子图

# 2. 在主图中使用子图作为节点
main_builder = StateGraph(State)
main_builder.add_node("subgraph_node", subgraph)  # 子图作为节点
main_builder.add_edge("start", "subgraph_node")
main_builder.add_edge("subgraph_node", END)

main_graph = main_builder.compile()
```

---

## 四、并行任务处理

### 4.1 基本结构

以 `START` 内置节点为起点，直接连接多个并行节点，最终汇总到一个 summary 节点。

```
                    ┌─────────────┐
                    │    START    │
                    └──────┬──────┘
                           │
           ┌───────────────┼───────────────┐
           ▼               ▼               ▼
      ┌─────────┐     ┌─────────┐     ┌─────────┐
      │ Node A  │     │ Node B  │     │ Node C  │
      └────┬────┘     └────┬────┘     └────┬────┘
           │               │               │
           └───────────────┼───────────────┘
                           ▼
                    ┌─────────────┐
                    │   Summary   │
                    └─────────────┘
```

### 4.2 并发状态合并冲突处理

| 方案 | 说明 |
|------|------|
| **方案一** | 不让多个并行节点修改同一个状态字段，每个节点对应一个独立的键 |
| **方案二** | 自定义一个合并函数，处理多个节点的输出合并 |

```python
# 方案一示例：每个并行节点写入不同的键
class State(TypedDict):
    query: str
    result_a: NotRequired[str]  # Node A 写入
    result_b: NotRequired[str]  # Node B 写入
    result_c: NotRequired[str]  # Node C 写入

# 方案二示例：自定义合并函数
from langgraph.graph import StateGraph

def merge_results(left: dict, right: dict) -> dict:
    """自定义合并逻辑"""
    merged = left.copy()
    merged.update(right)
    return merged

builder = StateGraph(State)
# ... 添加节点和边
```

---

## 五、循环逻辑与迭代优化

循环逻辑就是在任务节点完成之后，通过一个**路由函数**判断是否需要继续循环，需要就继续，不需要就退出循环。

```python
def should_continue(state: State) -> str:
    """路由函数：判断是否继续循环"""
    if state.get("iteration", 0) < state.get("max_iterations", 3):
        return "continue"   # 继续执行
    return "end"            # 结束循环

# 构建循环结构
builder.add_conditional_edges(
    "task_node",
    should_continue,
    path_map={"continue": "task_node", "end": END}
)
```

```
┌─────────┐      ┌─────────┐      ┌─────────────┐
│  START  │─────▶│  Task   │─────▶│  should_    │
└─────────┘      │  Node   │      │  continue?  │
                 └────┬────┘      └──────┬──────┘
                      │                  │
                      │         ┌────────┴────────┐
                      │         │                 │
                      │         ▼                 ▼
                      │    ┌─────────┐      ┌─────────┐
                      └────│continue │      │   end   │───▶ END
                           └─────────┘      └─────────┘
```

---

## 六、Human-in-the-loop（人机协作）

LangGraph 中的 `stream` 会逐步执行每个节点，并返回中间结果。

### 6.1 基本用法

```python
# stream 逐步执行，返回完整的 state 快照
for step in graph.stream(initial_state):
    print(step)  # 每个 step 是当前的完整状态
```

### 6.2 中断与恢复

在整个 `for` 循环中，通过读取 `step` 中参数的变化，来判断是否需要中断。然后可以通过 `app.invoke()` 来二次运行。

```python
for step in graph.stream(initial_state):
    # 检查是否需要人工介入
    if step.get("needs_human_review"):
        print("需要人工审核，当前结果：", step.get("result"))
        
        # 获取人工输入
        human_input = input("请输入反馈（通过/修改）: ")
        
        # 更新状态后继续
        new_state = {**step, "human_feedback": human_input}
        result = graph.invoke(new_state)  # 二次运行
        break
```

### 6.3 中断机制

中断机制同理——在路由函数或节点中设置中断条件，外部捕获后处理，再通过 `invoke` 恢复执行。

---

## 七、架构模式对比总结

| 模式 | 结构特点 | 适用场景 | 复杂度 |
|------|----------|----------|--------|
| **Supervisor** | 星型结构，中央协调 | 任务可拆分、需统一管理 | ⭐⭐⭐ |
| **Sequence** | 线性流水线 | 流程固定、顺序执行 | ⭐ |
| **Peer-to-peer** | 环形/网状结构 | 灵活自主、动态响应 | ⭐⭐⭐⭐ |
| **并行处理** | 分叉-合并结构 | 多任务并发执行 | ⭐⭐ |
