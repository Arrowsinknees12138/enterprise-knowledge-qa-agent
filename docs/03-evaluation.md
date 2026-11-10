# 评估与复现

2026 年 11 月 11 日在 Windows / Python 3.13.2 上运行。单元测试 10/10，Mock CLI 验收 8/8，真实 `qwen-plus` CLI 验收 8/8。[单测输出](../examples/unit-tests.txt) · [Mock 验收输出](../examples/mock-evaluation.txt) · [真实模型验收输出及用量](../examples/live-evaluation.txt)

## 真实模型验收

每个输出文件都是 CLI 的实际 JSON，含最终结果、模型接口请求数、令牌用量和逐步 Trace。Trace 中的 `next_action` 从实际执行顺序生成。

| 场景 | 输入 | 预期与实测 | 输出与 Trace |
| --- | --- | --- | --- |
| 明确答案 | [01](../examples/inputs/01-answer.json) | `answered`，引用 D03/P1 的 2 个工作日 | [01](../examples/live/01-answer.json) |
| 文档冲突 | [02](../examples/inputs/02-conflict.json) | `conflict`，读取 30 天和 15 天的原文后补查，给出双方引用和待确认项 | [02](../examples/live/02-conflict.json) |
| 资料外 | [03](../examples/inputs/03-outside.json) | `insufficient_evidence`，检索无命中，解释缺口 | [03](../examples/live/03-outside.json) |
| 同义改写 | [04](../examples/inputs/04-paraphrase.json) | `answered`，将刷卡进楼对应到门禁开通，引用 D03/P1 | [04](../examples/live/04-paraphrase.json) |
| 上下文追问 | [05](../examples/inputs/05-follow-up.json) | `answered`，从历史确定门禁主题，再检索和读取 D03/P1 | [05](../examples/live/05-follow-up.json) |
| 工具失败 | [06](../examples/inputs/06-tool-failure.json) | 注入 `read_doc` 故障后，`tool_error` 保留工具观察和证据缺口 | [06](../examples/live/06-tool-failure.json) |
| 更换原文 | [01](../examples/inputs/01-answer.json) | 仅把 D03/P1 的 2 个工作日改成 4 个工作日，结果引用新原文 | [07](../examples/live/07-updated-source.json) |
| 执行上限 | [01](../examples/inputs/01-answer.json) | 限制为 1 步，返回 `step_limit` 和未完成说明 | [08](../examples/live/08-step-limit.json) |

正常路径读到 D03 原文后结束。冲突路径读完两个期限后执行新的检索。资料外问题则在空结果后调整查询。可按 Trace 的 `arguments`、`observation` 和 `next_action` 核对这些差异。

## 开发中发现的问题

首轮真实模型识别了两个期限，但跳过了补查，并在缺口中断言两份资料现行有效，验收脚本因此失败。修正后，冲突结束前须在读完双方后成功执行不同查询，最终缺口使用受控文本。在检索结果中加入未读文档 ID 后，本轮冲突用例从 8 步降为 5 步，省去两次重复读取和一次额外检索。[首轮冲突输出](../examples/initial-conflict.json) · [提交版冲突输出](../examples/live/02-conflict.json)

## 复现

```text
python -m unittest discover -s tests -v
python tests/run_examples.py --mode mock
python tests/run_examples.py --mode live
```

最后一条命令需要按 [README](../README.md) 配置模型密钥，并会重新生成 `examples/live/` 中的输出。验收脚本用临时知识库文件测试替换原文。Mock 的冲突用例预期为 `needs_clarification`，只验证多个候选可被读取、引用和交给用户确认。

## 限制

评估覆盖四份合成文档及两类可注入的本地工具故障，其中真实模型用例注入了读取失败。词表外表达可能漏检，宽泛词可能带来无关候选。引用校验保证逐字可溯源，证据与问题的语义匹配以及冲突判断仍取决于模型。不同模型或重跑可能选用不同的查询和步骤。
