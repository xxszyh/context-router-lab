# 离线回答质量评测：生成、判分、配对报告

本流程不发起任何网络或模型调用。它把上一轮 LongMemEval 检索结果接到
回答质量评测，让本地模型、外部运行器或人工判分都能使用同一组输入。
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

## 2. 导入回答

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

只有完整的回答集合能进入判分。判分请求按固定 seed 打乱顺序，只含不透明 ID、
回答哈希及题目、参考答案、候选回答组成的提示，不暴露方案名称或生成模型名。
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

## 4. 生成配对质量报告

~~~sh
ctxlab longmemeval-answer-score .local/answer-plan/plan.private.json .local/answers.jsonl .local/judgments.jsonl .local/answer-quality.json
~~~

计划内容及输入/回答哈希、缺题、重复或未知 ID、混用模型/生成配置/判分模型、
非法布尔值都会拒绝导入。每个方案在同一题目集上计分，不能静默删掉失败题。
报告输出按方案及题型的参考答案判分准确率、记忆用量、截断/失败比例，
以及 joint 相对其余方案的精确 McNemar 和 20,000 次配对 bootstrap 区间。
这些多重比较标为探索性，不能仅挑显著的那一项作结论。

本阶段只覆盖检索报告中的**可回答题**。尚不评价拒答、不宣称样本独立留出、
不自动确立生产可用性。真实回答、独立判分、预先声明的主比较及拒答实验
是下一步的数据工作。本轮只把这条可复核的链路接通，没有生成新的质量成绩。
