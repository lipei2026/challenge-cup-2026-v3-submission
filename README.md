# V3 自适应推理数学智能体

本仓库保存用于正式测评的 `V3-AdaptiveCompute-Submission` 数学智能体。它是一个可以独立提交的
最小运行仓库，不包含本地数据集、参考答案、实验输出或评测日志。

## 评测入口

评测平台从仓库根目录的 `user_agent.py` 加载智能体：

```python
from user_agent import ReasoningAgent

agent = ReasoningAgent(client=official_client)
result = agent.solve(problem, metadata)
```

入口要求平台传入官方 Intern-S client。提交入口只向冻结 V3 暴露官方承诺的
`client.chat(messages, temperature, max_tokens)` 接口，不读取或修改 client 的内部字段。
`solve` 返回一个字典，其中包含非空字符串
`final_response` 和可以被 JSON 序列化的 `trace`。本仓库不保存 API Key，也不要求读取
本地密钥配置文件。

## V3 架构

V3 根据题目类型和推理状态动态分配计算量：

1. 简单题优先进入 Direct Solve，避免无条件执行长程推理。
2. 简单题的直接回答被截断时，可以进行一次独立短答案恢复。
3. 证明题或困难题进入基于子目标和已验证引理记忆的多轮推理。
4. 每轮根据新增引理、目标完成情况和具体推理缺口决定继续或停止。
5. 连续停滞时最多执行一次定向缺口修复，不会无限循环。
6. 最终答案经过过程验证；发现明确错误时最多修订一次。
7. 修订或部分最终调用失败时，尽可能保留此前完整且可判分的解答。
8. 未被内部节点捕获的超时或异常会在根入口触发一次应急直答，保证接口仍返回非空结果；入口只发起一次调用，重试和超时由平台 client 管理。

## 文件说明

- `user_agent.py`：比赛要求的根入口文件。
- `user_agent_v3.py`：V3 自适应计算、短答案恢复和推理预算控制。
- `user_agent_v2.py`：已验证引理记忆、最终综合和过程验证。
- `user_agent_adaptive.py`：V2/V3 继承的 Direct/Deep 题型路由。
- `user_agent_new.py`：共享的 LangGraph 状态、协议解析和基础工具函数。
- `llm_client.py`：本地调试使用的兼容 client；正式评测使用平台提供的 client。
- `requirements.txt`：运行所需的 Python 依赖。

其中 V3 及其继承链文件保持冻结；正式提交特有的顶层异常降级只位于 `user_agent.py`。

## 运行约束

- 运行时只使用官方 Intern-S client 和本地 Python 库。
- 不调用其他在线模型、外部 API、联网检索或在线 MCP。
- 不依赖绝对路径、隐藏测试数据、参考答案或非 Python 工具链。
- 正式评测入口不依赖平台 client 的私有 telemetry 方法。
- Python 版本需要为 3.10 或更高版本。

## 提交前检查

建议在干净的 Linux/Python 环境中安装 `requirements.txt`，并确认：

1. 可以从 `user_agent.py` 导入 `ReasoningAgent`。
2. `ReasoningAgent(client=official_client)` 可以正常初始化。
3. `solve(problem, metadata)` 返回非空 `final_response`。
4. 返回字典及 `trace` 可以被 JSON 序列化。
