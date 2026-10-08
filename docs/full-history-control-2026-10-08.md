# 完整历史对照与离线试跑，2026-10-08

跨渲染比较只能回答“相同检索结果怎样呈现更好”。本轮增加 `full_history` 对照，
直接比较全部历史与 joint 路由后的输入。**真实回答仍未生成，答案质量仍未测出。**
旧的 278 对、556 条跨渲染请求保留，未覆盖或改写。

## 对照含义

`--arm full_history` 必须单独导出。它读取源数据的全部唯一会话，保持会话输入顺序、
角色、原始发言内容与空白，不查询排序、不选片段、不按路由预算截断。
相同 ID、相同内容的重复会话仍遵守已有去重规则；有多个记录日期时保留全部不同日期，
不会挑选有利日期。生成输入没有参考答案、标注、拒答标签或原始会话 ID。

私有计划声明 `rendering=complete-history-v1`、
`memory_cap_policy=unbounded_complete_history`，每题记录源会话数。
`budget` 在两侧声明同一个计数器以及**候选方案**的上限。该上限不裁剪完整历史。
默认仍是原来的四个方案；不能把完整历史渲染挂到 joint 上，也不能把它混入四方案计划。

~~~sh
ctxlab longmemeval-answer-plan /path/to/longmemeval_s_cleaned.json .local/retrieval.json .local/full-pilot --arm full_history --limit 20 --seed 20261008
ctxlab longmemeval-answer-plan /path/to/longmemeval_s_cleaned.json .local/retrieval.json .local/routed-pilot --arm joint --rendering whole-turn-bm25-v3 --limit 20 --seed 20261008
ctxlab longmemeval-quality-plan .local/full-pilot/plan.private.json .local/routed-pilot/plan.private.json .local/full-comparison --comparison-kind full-history --seed 20261008
~~~

必须显式选 `--comparison-kind full-history` 才允许两侧选用不同会话。资料哈希、检索
报告、计数器、题目集、题型、参考、指令、拒答状态和日期歧义仍须匹配。
默认 `rendering` 模式仍要求选择的会话及顺序相同；旧 manifest 不需迁移。

完整历史是 baseline，joint 是 candidate；混排公开输入只含不透明 ID、哈希、指令与
提示。两侧仍须采用同一个固定生成模型及配置，再用另一个模型或固定人工协议盲评。
[已有生成、统一盲评和配对评分命令](quality-comparison-2026-10-07.md)可直接使用这份新比较。

## 失败对照不能构成质量优势

除了原有准确率区间、精确 McNemar、完成失败率、平均记忆用量及拒答回退条件，
完整历史比较还要求 `complete_history_baseline_completed=true`：baseline 的每条
可回答及拒答响应都正常完成且非空。超出上下文、截断、空回答或调用异常仍留在原
分母，按错误计分，同时阻止样本质量验收。报告不会删除失败题目再宣布获胜。

这会把“完整历史在该服务上无法执行”与“路由提高答案准确率”区分开。不能把完整
历史压缩至候选上限后还称为此对照，也不能生成后另挑能容纳的题目。
若需换更小数据或另一种对照，应先导出新设计并明确它回答的研究问题。

正常结束不能识别服务在**输入端**悄悄裁剪历史。运行前仍必须使用实际生成器的
tokenizer、聊天模板、上下文上限和输出预留检查容量，并确认服务拒绝超限而非自动
裁剪。字符数不能证明可执行性；当前报告保持 `generator_token_budget_verified=false`。

## 本机真实输入检查

从已有 278 道 answerable 检索记录中，按题号排序、固定 seed=20261008 打乱后取前
20 题，生成前固定。它们已来自分析过的数据，属于探索性试跑，不是独立留出集。
两侧各 20 条，共 40 条待生成请求，未按证据保留结果筛选题目。

| 输入检查 | 完整历史 | joint 整条发言路由 |
| --- | ---: | ---: |
| 题目数 | 20 | 20 |
| 平均记忆字符数 | 494,720.60 | 10,594.55 |
| 平均完整提示字符数，含指令 | 495,121.25 | 10,995.20 |
| 完整提示字符范围 | 480,092–515,885 | 9,686–12,096 |
| 全部提示字符合计 | 9,902,425 | 219,904 |

平均记忆字符比为 0.021415（约减少 97.86%），这只是输入大小，不是实际 token、
费用、吞吐或质量收益。完整历史的来源重建核验覆盖全部 20 题；18 道标注完整题
的全部标注发言均完整保留，2 道标注未知题仍在生成集合中。候选的同项诊断为
13/18，有 1 道检索丢失及 4 道渲染丢失；它们都保留在待测集合。

本机已检查已知模型缓存、命令和本地服务，尚未发现可直接使用的生成模型服务。
没有安装或下载生成模型，没有加载生成权重或占用 GPU。40 请求预览
`network_calls=0`、无回答输出；预览模型名是占位符，不代表选定了实际模型。

本机私有产物：

- `.local/answer-plan-full-history-pilot-2026-10-08/`：完整历史源计划。
- `.local/answer-plan-joint-turn-pilot-2026-10-08/`：候选源计划。
- `.local/quality-comparison-full-history-pilot-2026-10-08/`：冻结比较和混排请求。
- `.local/full-history-pilot-baseline-audit-2026-10-08.json`、候选 audit：逐条来源重建。
- `.local/full-history-pilot-preflight-2026-10-08.json`：字符统计、来源哈希、预览和容量未验证状态。

比较 manifest SHA-256：`d373edf7fa2306fec136d1c14c90af61892840201aaff772745a6a35f693a297`。
公开请求 SHA-256：`beaa88e1948b82b1acc9c0886e2c0bcf6b582425334d1e030d0f9077ab1c3e5a`。
来源审计单独运行，因为跨渲染审计故意要求同方案、同会话：

~~~sh
ctxlab longmemeval-render-audit /path/to/longmemeval_s_cleaned.json .local/full-comparison/baseline.plan.private.json .local/baseline-audit.json
ctxlab longmemeval-render-audit /path/to/longmemeval_s_cleaned.json .local/full-comparison/candidate.plan.private.json .local/candidate-audit.json
ctxlab longmemeval-run .local/full-comparison/generation.requests.jsonl .local/full-comparison/answers.jsonl --model dry-run-placeholder-2026-10-08 --max-requests 40
~~~

最后一条默认只预览。当前没有完成模型容量预检、真实生成、可信独立判分或独立数据
复验，也没有付费调用。即使探索性样本条件以后通过，`default_promotion_eligible`
仍保持 false，不能据此调整默认路由。原始资料、私有计划和全部实验输出留在 `.local/`。

本轮 410 项测试、Ruff 检查和格式、严格 mypy、wheel 构建及从 wheel 导入/运行命令
检查通过。新增测试覆盖未截断历史、来源完整性、显式控制模式、混排配对、基线失败
及拒答分母；测试响应是虚构 fixture，不是答案质量成绩。
