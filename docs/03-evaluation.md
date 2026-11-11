# 评估与复现

2026 年 11 月 11 日在 Windows / Python 3.13.2 上运行。单元测试 29/29，Mock CLI 验收 12/12。常规回归在代码提交 `091ed8d` 上通过 19/19。DashScope 日志保存早期版本的 10 组真实模型记录。[单测输出](../examples/unit-tests.txt) · [Mock 验收输出](../examples/mock-evaluation.txt) · [DeepSeek 常规回归](../examples/deepseek-evaluation.txt) · [早期 DashScope 验收](../examples/live-evaluation.txt)

## 50 题场景评估

修复后在 `6585e8d` 上用 DeepSeek 官方 API 完整跑过 50/50，28/28 道应引用题精确匹配，39/39 道有相关资料的题首轮 Top 3 至少命中一份，未观察到无依据作答。S31 和 S38 遇到连接类错误，各自动重跑一次后通过。题集和评分口径与 49/50 的复评分相同。[最新完整实跑](../examples/scenarios-50/fixed.json)

在原有 30 题上增加 S31 至 S50。新增题和评分器在实跑前提交为 `5a432ff`。题集包括跨文档双问、适用人群、日期与计日细节、原文改动、上下文和执行预算。评分检查正确的引用、所需的原文读取、答案值和运行状态。报告记录两个题集文件和合并后的 SHA-256。[原有 30 题](../examples/scenarios-30/cases.json) · [新增 20 题](../examples/scenarios-50/additional-cases.json)

DeepSeek 官方 API 完成 50 题全量运行，其中两题遇到连接类错误并各重跑一次。按首轮冻结的标注通过 48/50。S45 的原标注把“通过服务台生成的链接”当成了明确的起算规则，原文只给出链接失效时长。修订标注为资料不足，并要求实际读取 D12。用修订后的标准复核同一批 Trace，总分为 49/50，新增题 20/20。[首轮完整 Trace](../examples/scenarios-50/baseline.json) · [修订后的复评分](../examples/scenarios-50/review.json)

复评分中，需要作答或报告冲突的 28 题有 27 题精确匹配应引用的片段。首轮检索的前三条在 39/39 道有相关资料的题中至少命中一份，在 36/39 道题中覆盖所有标注相关文档。四道跨文档题、三道语料改动题和八道执行边界题均通过。S41 与 S42 只改变原文中的有效期，回答由 15 分钟变为 45 分钟。

全量运行中，S04 读取了写明“提交完整的门禁权限申请后”的 D03/P1，仍返回资料不足。定向复测复现了 S04 的误判，并验证 S45 返回资料不足且实际读取了 D12。先前的 30 题运行中，S04 曾通过，也曾失败。[两题定向复测](../examples/scenarios-50/focused-retest.json)

修复后，S04 的模型仍提交 `insufficient_evidence`，执行器核对原文中的起算条件、工作日期限和适用人群，返回 `answered`，并记录 `correction`。外包人员时限、提交当天是否计入、重置链接起算点三个缺口题仍返回资料不足。单测还用修改后的四个工作日条款验证回答跟随实际原文。[修复后 Trace](../examples/scenarios-50/fixed.json)

设置 `DEEPSEEK_API_KEY` 后复跑全部题目。可用 `--case S42-reset-link-updated-value` 只验证修改后的重置链接时效。

```text
python tests/run_holdout.py --dataset examples/scenarios-30/cases.json --dataset examples/scenarios-50/additional-cases.json --output scenarios-50-replay.json
```

## 30 题场景评估

在首次调用模型前提交题集与评分器，冻结提交为 `ba9bf9d`。问题和可选历史会传给 CLI，标准答案由本地评分器使用。S22 用临时语料替换一条原文。故障与预算场景通过 CLI 的公开参数运行，评分还核对工具观察、退出码与用量。[题集和标准答案](../examples/scenarios-30/cases.json) · [基线报告](../examples/scenarios-30/baseline.json) · [首次复测](../examples/scenarios-30/first-retest.json) · [第二次复测](../examples/scenarios-30/second-retest.json)

| 场景 | 题量 | 基线通过 | 最终通过 |
| --- | ---: | ---: | ---: |
| 直接事实和多字段问答 | 6 | 5/6 | 6/6 |
| 同义改写、上下文切换和历史纠错 | 5 | 5/5 | 5/5 |
| 资料外和命中但所问字段缺失 | 6 | 6/6 | 6/6 |
| 缺对象、指代歧义和适用人群不明 | 3 | 3/3 | 3/3 |
| 冲突、替换原文、版本日期和适用范围 | 5 | 5/5 | 5/5 |
| 工具故障、步数与工具预算 | 5 | 5/5 | 5/5 |

基线唯一失败是 S04。首轮已检索到 D03，读取的原文明确写有“提交完整的门禁权限申请后”。模型正确复述了这句话，但把起算事件和具体的计日规则混在一起，最后返回 `insufficient_evidence`。提示词改为分别核对起算事件、期限长度和日历计数规则。原文已给出起算事件时，可以回答从何时开始。若用户追问当天是否计入，则单独核对是否有对应条款。

首次全量复测 S04 通过，S19 新出现误答。对话历史同时提到邮箱和云盘，问题未说清“那个文件大小上限”指哪项。模型根据自己生成的邮箱查询读取了两份文档，最后引用邮箱上限。引用可追溯，但对象未经用户确认。随后明确要求，多对象指代无法唯一确定时使用 `needs_clarification`，并在明确切换主题的用例中验证仍能直接回答。

第二次全量复测中 S19 正确追问，S04 再次误判。Trace 显示模型在找一条新员工专属的规定，虽然原文是面向员工的普通规则且未写例外。第三次改动明确了范围包含关系。用户属于原文所覆盖的人群时，可用普通规则回答，存在明示例外或询问特殊待遇时再查对应规则。

最终在 `e987607` 上一次完整运行通过 30/30。S04 引用 D03/P1，S19 先追问。应引用题的片段集合完全匹配从 16/17 到 17/17。首轮检索 Hit@3 前后均为 23/23，最终无依据或不适用的回答为 0。四份报告的题集和基础语料哈希相同。第二次复测记录了 4 次连接类重跑，最终一轮无需重跑。结果来自 AI 拟定的合成场景及同题对照，模型行为仍可能随运行变化。[最终报告和 Trace](../examples/scenarios-30/after.json)

设置 `DEEPSEEK_API_KEY` 后，可复跑全部 30 题或单独复跑 S04。

```text
python tests/run_holdout.py --dataset examples/scenarios-30/cases.json --output scenarios-30-replay.json
python tests/run_holdout.py --dataset examples/scenarios-30/cases.json --case S04-new-access-clock --output scenarios-30-s04.json
```

## 冻结对照评估

Codex 在首次调用模型之前拟定并提交了 12 道新问题及标准答案，提交为 `2367ed9`。评分器只将问题和可选历史传给 Agent。基线和改动后均使用 `deepseek-flash`，报告记录了相同的题集与语料 SHA-256，以及各自运行的代码提交。[冻结题集](../examples/holdout/cases.json) · [基线报告](../examples/holdout/baseline.json) · [改动后报告](../examples/holdout/after.json)

| 指标 | 改动前 | 改动后 |
| --- | ---: | ---: |
| 逐题通过 | 11/12 | 12/12 |
| 最终状态匹配 | 12/12 | 12/12 |
| 应引用题的片段集合完全匹配 | 6/7 | 7/7 |
| 首轮检索 Hit@3 | 11/11 | 11/11 |
| 首轮 Top 3 覆盖全部标注相关文档 | 11/11 | 11/11 |
| 无依据或混入不适用引用的回答 | 1/12 | 0/12 |

逐题通过要求状态、引用集合和必要追问符合标准。引用文字还需与 Trace 中实际读取的片段完全一致。Hit@3 指首次检索的前三名至少包含一份标注相关文档，分母为 11，排除语料中本就没有相关文档的健身房报销题。无依据回答计数包括 `answered` 状态错误或引用集合不符合该题标准的情况。

基线 H05 正确找到旧版和替代版，也在决策依据中判断出 2025 年 5 月适用旧规。最终却同时引用了旧规的 10 个工作日和 7 月才生效的 5 个工作日。提示词现在要求只引用直接回答所问字段、且适用于指定对象和日期的片段。仅用于排除其他版本的依据保留在 Trace。复测中 H05 只引用了 D08/P1，H06 询问新版生效日与期限时仍正确引用 D09/P1。[基线逐步记录](../examples/holdout/baseline.json) · [复测逐步记录](../examples/holdout/after.json)

设置 `DEEPSEEK_API_KEY` 后可复跑全组，或只复跑 H05。

```text
python tests/run_holdout.py --output holdout-replay.json
python tests/run_holdout.py --case H05-historical-rule --output holdout-h05.json
```

题集使用同一批合成文档，由 AI 拟定。改动后结果属于同题对照复测。结果说明本次改动纠正了观测到的引用问题，对其他未知问题的表现仍需另行评估。

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

评估器遇到连接或超时类错误时会重跑该用例一次，也可用 `--case` 定向复跑。重跑属于评估脚本，单次 Agent 请求的工具与步数预算独立计量。[常规回归日志](../examples/deepseek-evaluation.txt)

全量命令为 `python tests/run_examples.py --mode live --provider deepseek`，输出位于 `examples/deepseek/`。

## 限制

评估覆盖 12 份合成文档和两类本地故障。词表外表达可能漏检，宽泛词可能带来无关候选。当前会要求模型核对所有返回候选，这在多候选时会增加调用量。引用校验保证逐字可溯源，证据与问题的语义匹配以及冲突判断仍取决于模型。不同模型或重跑可能选用不同的查询和步骤。
