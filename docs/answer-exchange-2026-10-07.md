# 离线回答质量评测：生成、判分、配对报告

比较旧、新渲染策略时，使用新增[跨策略质量比较](quality-comparison-2026-10-07.md)：
固定一个主方案、混排两侧生成输入和判分请求，按题号统计最终答案差异。
本机 joint 比较已导出 278 对、556 条请求，全部待生成。下述完整多方案交换流程继续支持。

准备输入、导入和计分都不发起网络或模型调用。新增的本地批量运行器默认也
只预览，必须显式指定 `--execute` 才会请求本机服务。流程把上一轮 LongMemEval
检索结果接到回答质量评测，让本地模型、外部运行器或人工判分使用同一组输入。
尚未导入真实回答和独立判分时，状态始终为 `not_measured`。

## 1. 准备回答输入

先取得完整检索报告，再运行：

~~~sh
ctxlab longmemeval-answer-plan /path/to/longmemeval_s_cleaned.json .local/joint.json .local/answer-plan
~~~

默认比较 hybrid、router、joint 和 query_only 四个方案。query_only 是无记忆对照。
报告缺少 joint 时，可重复指定 --arm hybrid --arm router --arm query_only。
--limit 20 通过固定 seed 抽取 20 题；不指定 limit 则准备报告中的全部题目。
题目已经参与过检索研究，不能把这个抽样称为独立留出集。

生成三个文件：

- generation.requests.jsonl：交给生成模型的输入，含不透明 request_id、
  prompt_sha256、instructions 和 prompt。生成时将 instructions 放入系统指令，
  prompt 放入用户输入，所有方案使用相同的模型和生成参数。
- plan.private.json：本机保存的完整计划，包含参考答案、方案映射和来源审计。
  **不要把此文件交给生成模型。**
- status.json：样本量、请求量、输入预算及 pending_generation 状态。

渲染保留问题日期、会话日期及发言角色。原始会话 ID 可能包含 answer 前缀，
因此输入使用 Memory 1 等中性名称；has_answer、参考答案和金标准会话集合
不进入生成输入。每条被选会话分配相同预算，用相同的 BM25 策略挑选片段。
这项渲染策略与先前只测会话召回的实验不同，必须作为新实验报告。

可选 `--rendering chronological-union-v2` 合并重叠窗口并保留会话内原顺序、原始空白，
在完整来源词边界上匹配预算。默认仍为原来的 equal-share-query-passages-v1。
可选 `--rendering whole-turn-bm25-v3` 则按完整发言做 BM25 选择，保留原顺序；
发言都放不下时回退到有界窗口，不读取证据标签或偏好某个发言角色。
各策略的来源审计、指标局限及离线比较命令见
[渲染后的证据保留审计](rendering-audit-2026-10-07.md)。审计读取标注只用于测量，
不会把标注交给生成模型，也不把保留率称为回答准确率。

官方 cleaned S 数据中有 32 个整数参考答案，导入时转为文字用于判分。
内容相同的重复会话可能带不同日期：正文去重，但保留所有不同日期并在输入中
标记 ambiguous，不选取最早/最晚值、不删除该题。status 和私人计划记录这类
问题，日期冲突须随后续质量结果一并披露。

默认 --memory-budget 12000 计量 **字符数**，所有记忆方案共用这个上限；
query_only 使用零记忆。实际使用量逐条记录，不会把字符预算写成模型 token 预算。
若有生成模型自己的本机 tokenizer.json，可启用准确的记忆 token 计量：

~~~sh
python -m pip install -e ".[token_budget]"
ctxlab longmemeval-answer-plan /path/to/data.json .local/joint.json .local/token-plan --memory-budget 4096 --tokenizer-file /path/to/tokenizer.json --tokenizer-model explicit-generator-version
~~~

该模式记录 tokenizer 文件 SHA-256，禁止其自带的截断或 padding 影响计数，
裁剪时保留原始大小写与 Unicode。导入回答时模型名必须与声明一致。
它只计量记忆正文，不含聊天模板、问题和模型特殊 token；报告不宣称已核实
服务端完整输入的 token 上限。API 返回的 input_tokens/output_tokens 可另外记录。

## 2. 本地批量执行与导入回答

先预览，不启动服务、不写结果文件、不加载模型：

~~~sh
ctxlab longmemeval-run .local/answer-plan/generation.requests.jsonl .local/answers.jsonl --model explicit-local-generator-version
~~~

预览显示输入 SHA-256、待处理量及本次调用/输出 token 上限。默认最多调用 20 次、
每次最多生成 256 token；这不是完整实验的预算，也不包含输入 token。
已有本机兼容服务时，可以选择执行一个小批次：

~~~sh
ctxlab longmemeval-run .local/answer-plan/generation.requests.jsonl .local/answers.jsonl --model explicit-local-generator-version --base-url http://127.0.0.1:11434/v1 --max-requests 20 --execute
ctxlab longmemeval-run .local/answer-plan/generation.requests.jsonl .local/answers.jsonl --model explicit-local-generator-version --base-url http://127.0.0.1:11434/v1 --max-requests 20 --resume --execute
~~~

运行器只接受 localhost、127.0.0.1 或 ::1 的 HTTP(S) 地址，不读取环境代理或
API key，不跟随重定向。使用兼容的 Chat Completions 请求，服务端返回的 model
必须与显式声明完全一致。模型名本身不能证明权重不可变，实验记录仍应写明
本机实际模型版本及量化方式。新接口默认使用 `max_completion_tokens`；
仅兼容旧实现时显式加 `--token-limit-field max_tokens`，不会静默切换参数。
参数规范见 [官方接口文档](https://developers.openai.com/api/reference/resources/chat/subresources/completions/methods/create)。

每次尝试写入 `answers.jsonl.checkpoint.jsonl` 并 flush/fsync；结果文件按最新尝试
生成唯一 ID 的快照。续跑校验输入字节哈希、模型、端点、生成参数和超时配置，
跳过已记录的成功与失败。仅本次批量上限可以调整。失败不自动重试；明确重试
使用相同命令加 `--resume --retry-failures --execute`。所有方案最终必须采用同一
重试政策，并在实验记录中披露重试。请求错误保存为空回答及 error 状态，仍在分母中；
空或截断的生成也标为失败，保留原始文本、结束原因及已知用量，允许显式重试。
未写完的 UTF-8 尾行仅在执行续跑时修复；已完成记录损坏则拒绝读取。

同一输出只运行一个进程。已经落盘的请求不会重复调用；若进程在服务已响应、
记录尚未落盘时被强制终止，续跑可能重复这一条，不能保证服务调用恰好一次。
检查点保留尝试历史。报告中的用量与延迟对应最终回答，不代表重试累计消耗；
缺失用量保持 null。单次 HTTP 超时最多 60 秒。

外部运行器为每个请求写一行 JSON。下面是**协议示例，不是实验结果**：

~~~json
{"request_id":"从输入复制","prompt_sha256":"从输入复制","hypothesis":"模型的最终回答","model":"explicit-generator-version","generation_config":{"temperature":0,"max_output_tokens":256},"stop_reason":"stop","input_tokens":null,"output_tokens":null}
~~~

generation_config 必须包含运行器实际采用的全部生成参数；所有方案的模型和
配置必须相同。stop_reason 必填，成功值为 stop、end_turn 或 completed。
长度截断、空回答及其他未完成响应不会计为正确，即使判分器说正确。
原始回答保留，重判无需再次生成。没有供应商用量报告时使用 null，不能假装为零。

## 3. 准备独立判分

~~~sh
ctxlab longmemeval-judge-plan .local/answer-plan/plan.private.json .local/answers.jsonl .local/judge.requests.jsonl
~~~

只有完整的回答集合能进入判分。判分请求按固定 seed 打乱顺序，包含不透明 ID、
回答哈希、完成/空回答标志及题目、参考答案、候选回答组成的提示，不暴露方案名称或生成模型名。
独立模型或人工判分应采用一致标准，不得利用方案映射。

判分规则参考 [LongMemEval 官方评价代码](https://github.com/xiaowu0162/LongMemEval/blob/main/src/evaluation/evaluate_qa.py)：
多会话题需要完整信息；时间计数允许相差一；偏好题需要正确运用用户偏好，
不要求覆盖 rubric 每一点。本项目使用自己的提示措辞和严格导入格式，
不能将任意模型或人工结果称为官方模型评测。

每个判分输出一行，correct 必须是真正的 JSON 布尔值：

~~~json
{"request_id":"从判分输入复制","answer_sha256":"从判分输入复制","correct":true,"judge_model":"explicit-judge-version"}
~~~

人工判分可使用明确的版本化协议名作为 judge_model，并在实验记录中说明判分者。
模型若输出其他文字，不得用包含 yes 的子串判断；需重新判分或明确处理失败。

本地判分也可使用同一个运行器，指定独立判分模型：

~~~sh
ctxlab longmemeval-run .local/judge.requests.jsonl .local/judgments.jsonl --stage judge --model explicit-local-judge-version
~~~

预览之后，显式加 `--execute` 分批执行，再用 `--resume` 续跑。只接受正常结束的
完整 yes/no 输出（忽略前后空格和大小写）；其他输出记录失败，不生成真假判分，
因此完整计分会拒绝缺失判分。已失败、截断或空的生成回答按既定政策计零，不调用
判分模型，并明确写入 `judgment_origin=failed_generation_policy` 和
`judge_call_performed=false`；这不是模型判分。所有模型判分必须使用相同的配置。

## 4. 生成配对质量报告

~~~sh
ctxlab longmemeval-answer-score .local/answer-plan/plan.private.json .local/answers.jsonl .local/judgments.jsonl .local/answer-quality.json
~~~

计划内容及输入/回答哈希、缺题、重复或未知 ID、混用模型/生成配置/判分模型、
非法布尔值都会拒绝导入。每个方案在同一题目集上计分，不能静默删掉失败题。
报告输出按方案及题型的参考答案判分准确率、记忆用量、截断/失败比例、可用的延迟及 token 用量，
以及 joint 相对其余方案的精确 McNemar 和 20,000 次配对 bootstrap 区间。
这些多重比较标为探索性，不能仅挑显著的那一项作结论。

## 5. 单独评测拒答

检索默认仍只统计可回答题。`--include-abstention` 可额外保存拒答题的检索输入，
`--abstention-only` 只处理拒答题。空金标准或以 `_abs` 结尾的题目不能计为
“检索成功”；它们的检索指标为 null，不进入 all-gold 统计。

~~~sh
ctxlab longmemeval /path/to/data.json .local/refusal-smoke.json --abstention-only
ctxlab longmemeval-answer-plan /path/to/data.json .local/refusal-smoke.json .local/refusal-plan --include-abstention --arm hybrid --arm router --arm query_only
~~~

上述默认 hash 嵌入器只做离线冒烟，不等同于已完成的 neural/joint 实验；两种
检索配置的结果不得混为同一实验。要比较同一组方案的拒答能力，在相同模型、
检索与生成配置下重新取得带拒答记录的报告（使用 `--include-abstention`）。

生成输入不泄露拒答标签。只有独立判分看到拒答参考解释，判断候选是否明确
承认信息缺失。空回答或执行失败不能代替正确拒答。质量报告将可回答题放在
summary/by_type/paired，将拒答题放在 abstention_summary、abstention_by_type 和
abstention_paired，分母独立。
仅拒答实验的主 summary 数量为零、准确率为 null，不制造综合分数。

本机已离线导出原 278 题、四个方案的 1,112 条请求，以及 18 道拒答题、三个
冒烟方案的 54 条请求。两者均未生成真实回答。现有题目不属于独立留出集，
质量提升、拒答能力及生产延迟仍未得到实验支持；下一步需要真实回答、独立
判分和预先声明的主比较。本轮没有新增回答质量成绩。
