# Part2 学习笔记

## 1. 动态示例选择

定制实例选择器是通过自己实现 `BaseExampleSelector` 的相关接口来实现的。

从 json 文件获取示例，直接使用或定制实例选择器，然后构建这个实例选择器之后，再构建整个 `FewShotPromptTemplate`，然后就可以直接用了。

### 动态生成提示词

`FewShotPromptTemplate` 类在执行 `format` 方法之后，就会调用 `example_selector` 的 `select_examples` 方法，然后这个 `select_examples` 方法的输入就是 `few_shot_prompt.format` 传入的内容。

---

## 2. 结构化输出

`8.py` 从第二步开始，先定义了这个 `ToolInfo`，设置好我要结构化输出什么，然后第三步创建解析器之后，`parser` 这个对象他就已经构建好要求模型输出结构化回答的提示词了，这个提示词里面的部分信息就是在 `ToolInfo` 中 `description` 中的内容。

`parser.get_format_instructions()` 会返回一段提示词，然后控制模型输出的结构，本质就是写提示词，只是他帮我们内置好了。

---

## 3. 解析模型的输出，自定义解析器

`BaseOutputParser` 子类通过 `get_format_instructions` 生成提示词，约束大模型的输出格式；然后 `parse` 方法依据该格式，将模型输出的文本解析成程序可直接使用的数据结构。

本质就是：**先约定格式，再按格式解析**

---

## 总结

这里主要讲了提示词模版，然后再到少量的示例选择器，之后就是输出解析，主要就是通过给模型一个提示词模版，然后模型根据这个模版生成回答，最后再通过解析器解析这个回答，变成程序可以直接使用的数据结构。
