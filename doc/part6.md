 python语法
class TaskState(TypedDict):   ，这个括号是继承的意思
节点函数里的 state: TaskState 只是 Python 类型注解，运行时没有任何效果。
动态类型（运行时确定）	函数不需要声明返回类型	类型注解：可选的，仅用于提示/工具	，运行时不检查注解，注解被忽略	
langgraph
langgraph中的状态可以理解为一块共享内存，在langgraph图中的每一个节点都可以访问和修改这个状态。
定义为
from typing import TypedDict, NotRequired

class TaskState(TypedDict):
    user_query: str #用户原始查询
    tool_result: NotRequired[str] #工具调用结果
    final_answer: NotRequired[str] #最终回答
    progress: NotRequired[int] #任务进度百分比

from pydantic import BaseModel, Field
from typing import Optional

class TaskState(BaseModel):
    user_query: str = Field(description="用户原始查询")
    tool_result: Optional[str] = Field(default=None, description="工具调用结果")
    final_answer: Optional[str] = Field(default=None, description="最终回答")
    progress: Optional[int] = Field(default=None, description="任务进度百分比")

StateGraph(TaskState)在创建图时直接放进去就行
节点函数不能直接修改状态，只能返回更新的部分。每个节点只负责"声明我要改什么"，由图引擎统一合并。

节点类型
llm，工具类，数据处理类