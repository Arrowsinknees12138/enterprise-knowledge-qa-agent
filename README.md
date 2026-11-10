# 企业知识问答 Agent

小影校招 AI Agent 开发考核第 3 题。员工可以询问合成的公司制度或产品 FAQ。CLI 返回任务状态、可核对的原文引用、资料缺口、待确认问题和执行 Trace。

## 环境与安装

需要 Python 3.10 或更高版本。在仓库根目录执行下列命令。项目仅用 Python 标准库，依赖清单为 [requirements.txt](requirements.txt)。

## 离线运行

```text
python -m src.knowledge_qa --mode mock --input examples/inputs/01-answer.json
python -m unittest discover -s tests -v
python tests/run_examples.py --mode mock
```

Mock 无需网络和环境变量。它用确定性规则跑同一套工具、引用校验和 Trace。Mock 遇到多份候选时展示原文并请求确认，用于验证执行流程。

## 真实模型运行

### DeepSeek

在 PowerShell 7 中运行以下命令。第一行提示输入 DeepSeek API Key，密钥仅设置在当前终端会话。默认模型是 `deepseek-flash`，可用 `DEEPSEEK_MODEL` 选择兼容模型。[DeepSeek API 文档](https://api-docs.deepseek.com/en/)

```powershell
$env:DEEPSEEK_API_KEY = Read-Host 'DeepSeek API Key' -MaskInput
python -m src.knowledge_qa --mode live --provider deepseek --input examples/inputs/01-answer.json
python -m src.knowledge_qa --mode live --provider deepseek --input examples/inputs/02-conflict.json
python tests/run_examples.py --mode live --provider deepseek
```

第一条问答预期返回 `answered`，引用 D03/P1 的 2 个工作日。冲突用例预期读取两条原文并补查后返回 `conflict`。JSON 中 `run.provider` 为 `deepseek`，`run.api_requests` 记录请求数，`trace` 记录实际步骤。全量验收会把 DeepSeek 输出写到 `examples/deepseek/`。

### DashScope

将 DashScope API 密钥设为 shell 环境变量 `DASHSCOPE_API_KEY`。默认使用 `qwen-plus` 和北京区兼容端点。其他兼容地址与模型名可通过 `QA_BASE_URL`、`QA_MODEL` 配置。变量说明见 [.env.example](.env.example)。CLI 从进程环境读取配置。[阿里云兼容接口文档](https://www.alibabacloud.com/help/en/model-studio/compatibility-of-openai-with-dashscope.md)

```text
python -m src.knowledge_qa --mode live --input examples/inputs/01-answer.json
python -m src.knowledge_qa --mode live --input examples/inputs/02-conflict.json
python -m src.knowledge_qa --mode live --input examples/inputs/06-tool-failure.json --fail-tool read_doc
python tests/run_examples.py --mode live
```

可用 `--question` 输入单条问题，用 `--input` 提供带可选 `history` 的 JSON，用 `--data` 替换知识库，用 `--max-steps` 调整决策预算（默认 8）。`--fail-tool search_docs` 或 `--fail-tool read_doc` 可模拟持续故障。未指定 `--mode` 时使用 Mock。

输出为 UTF-8 JSON。业务状态包括 `answered`、`conflict`、`insufficient_evidence` 和 `needs_clarification`，退出码为 0。`tool_error`、`model_error`、`step_limit` 的退出码为 1，输入或配置文件格式错误为 2。

## 评审入口

- [验收矩阵与复现方法](docs/03-evaluation.md)
- [DeepSeek 官方 API 实跑记录](examples/deepseek-evaluation.txt)
- [真实模型正常路径](examples/live/01-answer.json)和[冲突补查路径](examples/live/02-conflict.json)
- [真实模型工具失败路径](examples/live/06-tool-failure.json)和[更换文档原文](examples/live/07-updated-source.json)
- [修改原文后冲突消失](examples/live/09-resolved-conflict.json)和[检索命中但缺少答案](examples/live/10-missing-detail.json)
- [需求](docs/01-requirements.md)、[设计](docs/02-solution.md)、[AI 使用记录](docs/04-ai-usage.md)和[复盘](docs/05-retrospective.md)
