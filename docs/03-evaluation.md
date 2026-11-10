# 评估与复现

2026 年 11 月 11 日在 Windows / Python 3.13.2 上运行。单元测试 14/14，Mock CLI 验收 8/8。真实 `qwen-plus` 和 `deepseek-flash` 各有 10 组通过的实际执行记录。[单测输出](../examples/unit-tests.txt) · [Mock 验收输出](../examples/mock-evaluation.txt) · [DashScope 验收输出](../examples/live-evaluation.txt) · [DeepSeek 记录复核](../examples/deepseek-evaluation.txt)

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
| 修改原文消除冲突 | [02](../examples/inputs/02-conflict.json) | 仅把 D02/P1 的 15 天改为 30 天，模型读完两份原文后直接结束，给出 30 天和有效引用 | [09](../examples/live/09-resolved-conflict.json) |
| 命中但原文无法回答 | [01](../examples/inputs/01-answer.json) | D03/P1 只保留申请方式，仍检索命中且实际读取，返回 `insufficient_evidence` 和缺口，回答及引用为空 | [10](../examples/live/10-missing-detail.json) |

对照 02 与 09。问题、候选文档及检索词表保持一致，仅替换 D02/P1 的期限。原文冲突时，模型读完两份文档后补查适用依据并请求确认。期限一致时，模型在同一位置选择直接回答。对照 01 与 10 可核对同一候选命中后，答案如何取决于片段是否包含所问时长。[冲突 Trace](../examples/live/02-conflict.json) · [一致 Trace](../examples/live/09-resolved-conflict.json) · [缺字段 Trace](../examples/live/10-missing-detail.json)

检索对照使用同一个包含“刷卡进楼”的问题直接调用本地工具。仅保留“门禁”作为检索词时命中 0 份，启用别名后命中 D03。真实模型用例 04 则把该问法改写成门禁主题后取证。[对照测试](../tests/test_agent.py) · [改写 Trace](../examples/live/04-paraphrase.json)

## 开发中发现的问题

首轮真实模型识别了两个期限，但跳过了补查，并在缺口中断言两份资料现行有效，验收脚本因此失败。修正后，冲突结束前须在读完双方后成功执行不同查询，最终缺口使用受控文本。早期两次实跑中，加入未读文档 ID 前后，冲突路径由 8 步缩为 5 步，省去两次重复读取和一次额外检索。[首轮冲突输出](../examples/initial-conflict.json) · [提交版冲突输出](../examples/live/02-conflict.json)

## 复现

```text
python -m unittest discover -s tests -v
python tests/run_examples.py --mode mock
python tests/run_examples.py --mode live
```

最后一条命令需要按 [README](../README.md) 配置模型密钥，并会重新生成 `examples/live/` 中的输出。验收脚本用临时知识库文件测试替换原文。Mock 的冲突用例预期为 `needs_clarification`，只验证多个候选可被读取、引用和交给用户确认。

### DeepSeek 验证

DeepSeek 官方 API 首轮通过 9 组用例。第 10 组正确返回 `insufficient_evidence`，但将只说明申请方式的段落放进了引用列表，未通过验收。修复后，该状态的背景片段保留在 Trace，最终答案与引用均为空，并增加了回归单测。[首轮第 10 组输出](../examples/deepseek-initial-missing-detail.json)

修复后重跑时，资料外问题有一次模型连接或超时类错误，随后单独复测资料外问题与缺少答案两个场景，均通过。保存的 10 组 Trace 来自多次官方 API 实跑。冲突用例在读完两份原文后补查，改成同一时限后直接结束并引用原文。[冲突 Trace](../examples/deepseek/02-conflict.json) · [资料外 Trace](../examples/deepseek/03-outside.json) · [更换原文后的 Trace](../examples/deepseek/09-resolved-conflict.json) · [修复后第 10 组输出](../examples/deepseek/10-missing-detail.json)

全量命令为 `python tests/run_examples.py --mode live --provider deepseek`，输出位于 `examples/deepseek/`。

## 限制

评估覆盖四份合成文档及两类可注入的本地工具故障，其中真实模型用例注入了读取失败。词表外表达可能漏检，宽泛词可能带来无关候选。引用校验保证逐字可溯源，证据与问题的语义匹配以及冲突判断仍取决于模型。不同模型或重跑可能选用不同的查询和步骤。
