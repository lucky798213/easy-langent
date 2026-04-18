# Part3 学习笔记

## 1. LangChain 记忆

最后在执行的时候是通过 `RunnableWithMessageHistory` 这个组件来执行的，构成这个组件主要有 4 部分：

- **runnable**：就是要执行的执行链，通过管道 `|` 把组件连接起来
- **get_session_history**：获取 session 的方法
- **input_messages_key**：用户输入的内容要放到这个模版中的哪个 key 的上面
- **history_messages_key**：历史记录要放到这个模版中的哪个 key 的上面

```python
full_memory_chain = RunnableWithMessageHistory(
    runnable=base_chain,
    get_session_history=get_full_memory_history,
    input_messages_key="user_input",  # 请从输入字典中提取键名为 user_input 的值，作为用户的当前消息
    history_messages_key="chat_history"  # 传入提示词的历史消息键名
)
```

### RunnableWithMessageHistory 调用后内部的执行流程

在 `invoke` 方法中，输入有 `invoke({"user_input": "我叫小明，喜欢编程"}, config=config)`，程序从 config 中读取 sessionid，然后返回 `get_session_history` 读取到的历史记录，然后会调用这个历史记录对象的 `.messages` 属性，得到消息队列，然后就会构建一个字典 `merged_input`，这个字典有用户输入的键值和历史记录的键值，这些键都是在构建 `RunnableWithMessageHistory` 中就定义好的。然后就会将 `merged_input` 传递给 `base_chain`，然后就将 `merged_input` 映射到 `full_memory_prompt` 中，最终将格式化的完整的提示词发给 llm，得到结果，再将结果存储到调用 `get_full_memory_history` 得到的存储空间当中。

### 窗口记忆

窗口记忆就是通过 `get_full_memory_history` 来筛选历史对话的前几位。

### 摘要记忆

摘要记忆：就是在原来的基础上，对历史对话进行摘要，得到一个摘要，然后将摘要放到提示词中。

---

## 2. 自定义工具 Tool

首先就是定义参数模型，用于表示 tool 所需要的参数，里面还有对参数的描述：

```python
class TemperatureConvertInput(BaseModel):
    temperature: float = Field(description="需要转换的温度值，例如37.0")
    from_unit: str = Field(description="原始温度单位，只能是celsius或fahrenheit")
```

然后通过 `@tool` 装饰器传入参数模型，然后在 tool 装饰器下定义好这个工具的具体实现即可，最终构建 agent：

```python
agent = create_agent(
    model=llm,
    tools=tools,
    system_prompt=system_prompt,
    debug=True
)
```
“AI 调用 tool” 是错误的说法。AI 只输出调用建议，工具由你的代码执行。
工具结果存入 history 只是中间步骤，必须再次调用模型才能让模型基于结果生成最终回答。
---

## 3. Python 语法

装饰器 `@`，就是把 `@` 下面的函数当作一个变量传入给装饰器中，在装饰器中经过处理之后，返回一个新函数，然后原来的旧函数就变成了新函数。
