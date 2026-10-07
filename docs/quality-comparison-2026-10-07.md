# 跨渲染策略的回答质量比较，2026-10-07

检索召回和标注发言保留率不能代替答案准确率。新增流程把旧渲染与整条发言渲染
放入一个混排生成批次，再统一盲评、按题号配对。本机只导出输入和运行离线审计，
**没有生成真实回答，最终回答质量仍未测出**。

## 固定主比较

原先两个完整计划各有 278 题、四个检索方案、1,112 条请求。现在仅选择 joint
作为主比较，保留所有 278 题，每题生成旧、新输入各一次，共 **556 条请求**。
30 道缺少完整发言标注的题目仍进入答案评测；只有之前的证据指标需要排除它们。
六道日期歧义题也保留，不按输入审计的胜负筛选题目。

~~~sh
ctxlab longmemeval-quality-plan .local/legacy-plan/plan.private.json .local/turn-plan/plan.private.json .local/quality-comparison
~~~

两份源计划的资料、检索报告、预算、题目集、参考答案、拒答状态、选择的会话及
顺序必须匹配，只允许渲染策略不同。默认选 joint，可在生成前用 `--arm hybrid`
或 `--arm router` 创建另一份设计，不能生成后再挑最好方案。

导出的文件包括：

- `comparison.private.json`：主比较、题目集、来源哈希、统计 seed 和验收条件。
- `baseline.plan.private.json`、`candidate.plan.private.json`：主方案的完整私有计划。
- `generation.requests.jsonl`：混排公开输入，只含不透明 ID、提示哈希、指令和提示。
- `status.json`：导出时的状态快照；后续进度看运行器及最终报告。

参考答案、方案映射和来源请求 ID 只在私有文件中。即使两侧某题提示相同，也分配
不同 ID、保留两次生成机会，不去重或复用某次碰巧答对的回答。新 ID 属于这份比较，
原完整计划的回答不能直接导入。原计划不被修改。固定 seed 混排减小按方案分批的
偏差，不能保证模型服务或运行环境不漂移。

## 预览、生成与统一盲评

~~~sh
ctxlab longmemeval-run .local/quality-comparison/generation.requests.jsonl .local/quality-comparison/answers.jsonl --model explicit-local-generator-version
~~~

默认只预览，不构造客户端、不调用模型、不写回答。选定实际本地模型和完整参数后
才可显式执行；批量上限、断点恢复及失败重试沿用[已有运行器](answer-exchange-2026-10-07.md)。
两侧必须遵守相同重试政策。本轮没有执行生成、加载生成权重或使用 GPU。

设计冻结题目、主比较、验收条件及输入，**没有冻结尚未选择的生成模型和参数**，
manifest 明确记录 `generation_configuration_frozen=false`。导入会校验所有回答使用
相同模型与完整配置，保留实际配置和回答哈希。这不能证明外部运行器如实声明所有
参数或权重。如果计划使用本地 tokenizer，生成模型还必须匹配预算声明。

~~~sh
ctxlab longmemeval-quality-judge-plan .local/quality-comparison/comparison.private.json .local/quality-comparison/answers.jsonl .local/quality-comparison/judge.requests.jsonl
ctxlab longmemeval-run .local/quality-comparison/judge.requests.jsonl .local/quality-comparison/judgments.jsonl --stage judge --model explicit-independent-judge-version
~~~

第一条命令只导出，第二条默认仍只预览。完整回答集合才能进入判分。两侧候选混入
同一文件，判分器看到题目、参考答案和候选回答，看不到生成模型名、渲染策略或方案
字段。任务规则、严格 yes/no、失败生成计零均复用已有流程。

计分要求判分模型名与生成模型名不同，所有判分采用同一模型和配置。不同模型名
不能证明判分独立性或能力，仍需外部审查。人工判分也可导入，须提供固定协议：

~~~json
{"request_id":"复制输入 ID","answer_sha256":"复制输入哈希","correct":true,"judge_model":"human-panel-protocol-v1","judgment_config":{"protocol":"rubric-v1","panel":"independent-panel-v1"}}
~~~

这只是接口示例，不是成绩。缺少 `judgment_config` 的旧格式在旧计分命令仍受支持，
在跨策略比较中被拒绝。模型运行器自动记录实际判分配置。

## 配对计分与验收条件

~~~sh
ctxlab longmemeval-quality-score .local/quality-comparison/comparison.private.json .local/quality-comparison/answers.jsonl .local/quality-comparison/judgments.jsonl .local/quality-comparison/quality.json
~~~

按题号连接两侧，不依赖源计划或回答文件的行顺序。每个请求必须恰有一条回答和
判分，缺题、重复、未知 ID、过期哈希、配置混用均报错。空、截断或执行失败的回答
即使导入 true，也计错并留在分母。正常结束但为空的回答也计入失败完成率。

主统计是 candidate − baseline 的参考答案判分准确率差、配对胜负、精确 McNemar
和 20,000 次配对 bootstrap 95% 区间，seed 固定于 manifest。题型分组属次要探索。
导出时的默认样本验收条件为：

1. 精确 McNemar p ≤ 0.05，且准确率差的 bootstrap 下界严格大于 0。
2. 失败完成率不增加。
3. 平均记忆用量不增加，candidate/baseline ≤ 1。

导出前可用 `--minimum-accuracy-gain 0.05` 要求区间下界高于 5 个百分点，其他阈值
也可声明，包括 `--maximum-exact-mcnemar-p`。改变条件必须生成新设计，不能复用旧 manifest 哈希。各项判断和实测值
分别记录，不仅引用通过的条件。平均记忆单位来自预算，本机为**字符**，不等于
实际模型输入 token、完整聊天模板或费用。`generator_token_budget_verified=false`，
未知用量和延迟保持 null；最终回答行的用量不等于重试累计消耗。

拒答题保持独立分母，观察到拒答准确率下降会阻止样本验收；这不是拒答非劣效性
证明。仅拒答的比较不会制造可回答准确率或通过主验收。本机 joint 比较不含拒答，
此前 hash 嵌入的 18 题冒烟不能与 neural/joint 数据拼接。

## 当前离线产物和边界

本机目录 `.local/quality-comparison-joint-2026-10-07-v2/` 含 278 对、556 条待生成请求。
预览网络调用为 0，回答输出不存在。

- manifest SHA-256：`1b61f54d37eb090f92a836772a2c8cfa11a35e59a8cff9eab83672ae58095f6e`。
- 公开请求 SHA-256：`eb558220dedbf126d70941ee6a1c202269e9a09468f6c97a31ffcc5e1c59b2c0`。
- 原完整计划 SHA-256：`6927362ec48ed57572a69c3bfbdd3df5e7fb44a80932252155a4f390844457f9`。
- 整条发言源计划 SHA-256：`055655b8c96a9ae9f9f1719f510381e343e4e464fdcb9dfa332dc111227a8b81`。

真实源资料重建全部 556 条投影提示，与原输入完全一致。joint 严格证据保留率仍为
110/248 → 175/248，不因投影而改变，也不产生新的答案质量分数。审计命令为：

~~~sh
ctxlab longmemeval-render-audit /path/to/longmemeval_s_cleaned.json .local/quality-comparison/baseline.plan.private.json .local/quality-comparison/render-audit.json --compare-plan .local/quality-comparison/candidate.plan.private.json
~~~

这批题已用于诊断，重新划分不能把它变成独立留出集。哈希绑定产物、发现意外修改，
不能证明注册时间或抵抗重新计算全部哈希。即使日后样本条件通过，报告仍保持
`sample_is_held_out=false`、`comparisons_are_exploratory=true` 和
`default_promotion_eligible=false`。调整默认策略仍需固定生成器、可信独立判分及
独立数据上的完整实验。

本轮 389 项测试、Ruff、严格 mypy 均通过。测试回答和判分都是虚构 fixture，只验证
接口和统计规则；没有真实答案成绩或付费调用。私有计划及实验产物留在 `.local/`。
