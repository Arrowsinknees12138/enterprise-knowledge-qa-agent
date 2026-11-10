# 评估与复现

2026 年 11 月 11 日在 Windows / Python 3.13.2 上运行扩展版。单元测试 18/18，Mock CLI 验收 12/12，DeepSeek 官方 API 的完整一轮 CLI 验收 19/19。DashScope 日志保存上一版 10 组真实模型记录。[单测输出](../examples/unit-tests.txt) · [Mock 验收输出](../examples/mock-evaluation.txt) · [DeepSeek 验收输出](../examples/deepseek-evaluation.txt) · [上一版 DashScope 验收输出](../examples/live-evaluation.txt)

## 真实模型验收

每个输出文件都是 CLI 的实际 JSON，含最终结果、模型接口请求数、令牌用量和逐步 Trace。Trace 中的 `next_action` 从实际执行顺序生成。

| 场景 | 输入 | 预期与实测 | 输出与 Trace |
| --- | --- | --- | --- |
| 明确答案 | [01](../examples/inputs/01-answer.json) | `answered`，引用 D03/P1 的 2 个工作日 | [01](../examples/deepseek/01-answer.json) |
| 文档冲突 | [02](../examples/inputs/02-conflict.json) | `conflict`，读取 30 天和 15 天的原文后补查，给出双方引用和待确认项 | [02](../examples/deepseek/02-conflict.json) |
| 资料外 | [03](../examples/inputs/03-outside.json) | `insufficient_evidence`，解释缺口 | [03](../examples/deepseek/03-outside.json) |
| 同义改写 | [04](../examples/inputs/04-paraphrase.json) | `answered`，将刷卡进楼对应到门禁开通，引用 D03/P1 | [04](../examples/deepseek/04-paraphrase.json) |
| 上下文追问 | [05](../examples/inputs/05-follow-up.json) | `answered`，从历史确定门禁主题，再检索和读取 D03/P1 | [05](../examples/deepseek/05-follow-up.json) |
| 读取失败 | [06](../examples/inputs/06-tool-failure.json) | 注入 `read_doc` 故障后，`tool_error` 保留工具观察和证据缺口 | [06](../examples/deepseek/06-tool-failure.json) |
| 更换原文 | [01](../examples/inputs/01-answer.json) | 仅把 D03/P1 的 2 个工作日改成 4 个工作日，结果引用新原文 | [07](../examples/deepseek/07-updated-source.json) |
| 决策轮次上限 | [01](../examples/inputs/01-answer.json) | 限制为 1 步，返回 `step_limit` 和未完成说明 | [08](../examples/deepseek/08-step-limit.json) |
| 修改原文消除冲突 | [02](../examples/inputs/02-conflict.json) | 仅把 D02/P1 的 15 天改为 30 天，读完两份后直接结束，给出 30 天和有效引用 | [09](../examples/deepseek/09-resolved-conflict.json) |
| 命中但原文无法回答 | [01](../examples/inputs/01-answer.json) | D03/P1 只保留申请方式。实际读取后返回 `insufficient_evidence`，回答及引用为空 | [10](../examples/deepseek/10-missing-detail.json) |
| 相似候选干扰 | [11](../examples/inputs/11-attachment-noise.json) | 邮箱与云盘均命中，核对后只引用邮箱 D04/P1 的 25 MB | [11](../examples/deepseek/11-attachment-noise.json) |
| 多片段引用 | [12](../examples/inputs/12-multipassage.json) | 邮箱 FAQ 有三条原文，只引用回答回收站保留期的 D04/P2 | [12](../examples/deepseek/12-multipassage.json) |
| 明示替代关系 | [13](../examples/inputs/13-explicit-supersession.json) | 读取旧版 10 个工作日和替代版 5 个工作日，引用明确写有替代条款的 D09/P1 | [13](../examples/deepseek/13-explicit-supersession.json) |
| 适用人群 | [14](../examples/inputs/14-audience-scope.json) | 分清正式员工和外包人员，引用外包规则 D11/P1 | [14](../examples/deepseek/14-audience-scope.json) |
| 含糊追问 | [15](../examples/inputs/15-ambiguous.json) | 无历史时明确请求补充所问事项，返回 `needs_clarification` | [15](../examples/deepseek/15-ambiguous.json) |
| 检索失败 | [01](../examples/inputs/01-answer.json) | 注入 `search_docs` 故障，返回 `tool_error` | [16](../examples/deepseek/16-search-failure.json) |
| 业务工具预算不足 | [01](../examples/inputs/01-answer.json) | 限制为 1 次工具调用。检索执行，读取被拦截，返回 `tool_limit` | [17](../examples/deepseek/17-tool-limit.json) |
| 恰好用完工具预算 | [01](../examples/inputs/01-answer.json) | 限制为 2 次。检索和读取完成后预算为零，仍可引用 D03/P1 回答 | [18](../examples/deepseek/18-finish-at-limit.json) |
| 其他产品 FAQ | [19](../examples/inputs/19-password.json) | 从 D12/P1 回答密码重置链接 15 分钟失效 | [19](../examples/deepseek/19-password.json) |

对照 02 与 09。问题、候选文档及检索词表保持一致，仅替换 D02/P1 的期限。原文冲突时，模型读完两份文档后补查适用依据并请求确认。期限一致时，模型在同一位置选择直接回答。对照 01 与 10 可核对同一候选命中后，答案如何取决于片段是否包含所问时长。[冲突 Trace](../examples/deepseek/02-conflict.json) · [一致 Trace](../examples/deepseek/09-resolved-conflict.json) · [缺字段 Trace](../examples/deepseek/10-missing-detail.json)

检索对照使用同一个包含“刷卡进楼”的问题直接调用本地工具。仅保留“门禁”作为检索词时命中 0 份，启用别名后命中 D03。用例 11 的真实检索同时命中邮箱和云盘，得分分别为 2.5 和 0.5，最终只引用邮箱条款。缓存单测中模型请求检索两次、读取两次，底层各执行一次，剩余预算依次为 1、1、0、0，最后仍能结束。[对照测试](../tests/test_agent.py) · [干扰项 Trace](../examples/deepseek/11-attachment-noise.json)

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

前一版重跑时，资料外问题曾遇到模型连接或超时类错误。现有评估器遇到连接类错误会把该用例重跑一次，也可用 `--case` 定向复跑。该重跑属于评估脚本，单次 Agent 请求的工具与步数预算独立计量。扩展版本次完整一轮通过 19/19。[本轮实跑日志](../examples/deepseek-evaluation.txt)

全量命令为 `python tests/run_examples.py --mode live --provider deepseek`，输出位于 `examples/deepseek/`。

## 限制

评估覆盖 12 份合成文档和两类本地故障。词表外表达可能漏检，宽泛词可能带来无关候选。当前会要求模型核对所有返回候选，这在多候选时会增加调用量。引用校验保证逐字可溯源，证据与问题的语义匹配以及冲突判断仍取决于模型。不同模型或重跑可能选用不同的查询和步骤。
