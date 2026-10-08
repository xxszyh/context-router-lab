# 生成前的聊天模板容量核验，2026-10-08

本轮补上 `longmemeval-capacity-plan` 及运行器的 `--capacity-report`。
完整历史不能只看字符预算：现在使用本地分词器的实际聊天模板，逐条计算 system、
user 和 assistant 起始标记的 token，加上预留输出后核对声明的上下文容量。
**这不是质量成绩；没有生成真实回答或调用付费模型。**

## 离线容量计划

~~~sh
python -m pip install -e ".[chat_budget]"
ctxlab longmemeval-capacity-plan .local/comparison/generation.requests.jsonl .local/capacity.json --tokenizer-directory /path/to/local/tokenizer --model explicit-model-version --context-window 131072 --max-output-tokens 256
~~~

tokenizer-directory 必须已有 tokenizer_config.json、分词器文件和明确的聊天模板。
命令只从本机加载分词器，不下载权重、不加载模型、不连接推理服务、不启用远程代码。
无模板就报错，没有猜测格式的回退。设置 `tokenize=True`、`add_generation_prompt=True`、
`truncation=False`、`padding=False`；也关闭分词器文件中存储的截断/填充设置。
按照 [Hugging Face 的聊天模板说明](https://huggingface.co/docs/transformers/main/en/chat_templating)，
直接对模板分词，避免再次加入已经在模板里的特殊 token。

报告固定完整请求文件的字节哈希、每个请求的内容哈希、模型名、输出预留、上下文
容量、分词器文件哈希、模板哈希与 Transformers 版本。预检和执行使用同一个请求
校验器及消息构造函数。模型服务若使用不同模板或选项，需先统一配置再计数。
当前使用 tokenizer 的默认模板选项；不据此推断服务的 thinking 等选项已经匹配。

报告没有原始提示、参考答案或回答，只含不透明 ID、哈希与计数。输出不得覆盖请求，
也不得写进分词器目录。发现任何超限请求仍保存完整报告，状态为 blocked_context_overflow，
命令退出码为 1；不截断历史、不删除超限请求。

## 执行前阻断与服务用量核对

~~~sh
ctxlab longmemeval-run .local/comparison/generation.requests.jsonl .local/comparison/answers.jsonl --model explicit-model-version --capacity-report .local/capacity.json --max-output-tokens 256
~~~

默认只预览。使用 `--execute` 前会核对报告与整份请求、模型、输出预留及本地分词器
文件一致。只要有一条超限，就在构造客户端、创建输出和 checkpoint 之前停止。
即使 --max-requests=1，后续未排入本轮的超限请求也阻止执行，不能隐去它们。

执行时还要求服务返回的 input_tokens 与该请求的预检计数相等。缺失或不符时，将
该次响应记录为失败/空回答并停止后续批次，防止继续消耗调用。修正服务配置后，
可显式 `--resume --retry-failures`；容量计划哈希写入配置，恢复不能换另一份计划。
报告、请求和回答均保留。只有已失败/空生成的判分项走既有 policy-zero，不占模型
上下文或调用额度；它们仍在判分集合和失败分母中。

容量保护只对传入 --capacity-report 的运行生效。旧调用没有自动获得容量验证。
本地模板计数及声明上限不证明服务配置、权重身份、显存或速度；服务用量本身也依赖
其真实报告。容量计划保持 service_tokenization_verified=false 及
generator_token_budget_verified=false；成功计数不会自动变成质量或性能验收。

## 20 对真实输入的容量结果

只为计数下载了 Qwen/Qwen3-4B 的三个文件，约 11.4 MB：config.json、tokenizer.json、
tokenizer_config.json，固定 revision=`1cfa9a7208912126459214e8b04321603b3df60c`。
未下载 safetensors/GGUF，未加载生成权重。探测模型名是占位标识，尚未选定生成器。

[官方模型说明](https://huggingface.co/Qwen/Qwen3-4B)给出原生 32,768 token 和使用 YaRN
扩展至 131,072 的设置。分别作为本次容量假设，均预留 256 输出 token：

| 20 题输入 | 完整历史 | joint 整条发言路由 |
| --- | ---: | ---: |
| 最少完整输入 token | 104,348 | 2,069 |
| 最多完整输入 token | 108,031 | 2,864 |
| 平均完整输入 token | 106,414.90 | 2,506.65 |
| 32,768 容量假设下超限项 | 20/20 | 0/20 |
| 131,072 容量假设下超限项 | 0/20 | 0/20 |

这回答了输入能否容纳的一个必要条件，没有回答模型能否保住质量。
对真实 40 请求文件运行 32k 阻断验证，execute=True 也没有构造客户端或创建回答目录；
131k 报告的预览 network_calls=0。原比较 manifest、40 请求及之前的 556 请求设计保留。

本机 nvidia-smi 报告 RTX 5070 Laptop GPU、8151 MiB 显存。固定配置为 36 层、8 个
KV 头、head_dim=128。普通、未量化 FP16 K/V 缓存的估计为：

~~~text
每个输入 token: 36 × 8 × 128 × 2(K/V) × 2字节 = 147,456 字节
完整历史平均缓存: 106,414.9 × 147,456 / 2^30 ≈ 14.61 GiB
路由输入平均缓存: 2,506.65 × 147,456 / 2^30 ≈ 0.344 GiB
~~~

这是从配置推导的普通缓存估计，排除权重、工作区及服务开销，并非实测峰值。
完整历史的该缓存已经超过显存，后续实际服务配置须解决缓存/内存与速度，再执行
真实生成及独立判分。131k 在长度上通过不能宣称本机推理已经通过。256 输出只是
容量探测预留，也没有被选定为最终质量实验的输出预算。

## 产物与验证

- 本机容量报告：`.local/full-history-capacity-qwen3-native-2026-10-08.json` 与 extended 对应文件。
- 原生报告 capacity_sha256：`3a9a6cdeb16f72b4504d18f119f1a40444e2e1e369c3720949cc2476ef266eff`。
- 扩展报告 capacity_sha256：`ab49375755c8d7923477ff4c7cc687073593d5be81ebaf1c8452dfabff890ba6`。
- 本机汇总：`.local/full-history-token-feasibility-2026-10-08.json`，
  SHA-256=`55aa706b0642d89af89d8ca792f5f6bc49d8b6a0874bba724d054345cc8a406d`。
- 缓存估计：`.local/qwen3-kv-memory-estimate-2026-10-08.json`，
  SHA-256=`f636bf1a24e3eece73d64338f3fca084b9a09b2c426e766c847818681b599d06`。

436 项完整测试、Ruff 检查/格式和严格 mypy（75 文件）通过。新增测试使用本地虚构
分词器，不下载模型：覆盖模板和输出预留、关闭存储截断、全请求阻断、过期计划、
变更分词器、服务计数不符后停止、显式恢复、判分 policy-zero 和保留预检证据。
CI 新增 chat_budget 依赖以确保这些用例实际运行。
