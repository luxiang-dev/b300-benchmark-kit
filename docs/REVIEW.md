**B300 测试项目审查 · 2026-10-11**

当前项目存在需要修复的离线依赖和证据校验问题。建议先处理分词器缓存及实际运行配置，再进行完整 DeepSeek-R1 / MLPerf v6.0 Offline 实测。该次审查未修改测试程序或公开基线；GitHub 仓库额外提供入口说明与本审查记录。

审查范围为本仓库，重点检查 DeepSeek 扩展，并检查公共报告、下载与基础启动流程。原有 56 项 CPU 单元测试通过，三个 Bash 脚本通过语法检查。这些结果不能确认 Docker 镜像或实际 B300 运行可用；本机没有相应 GPU、完整模型、数据集或 MLPerf runtime。

1. **[P1] 准确率阶段依赖的 Hugging Face 分词器缓存未准备、也未预检。**

   位置：[准确率环境预检](../deepseek_mlperf.py#L186)、[准备流程](../README_DeepSeek_MLPerf.md#L71)。

   下载器把原始分词器放在 `build/models/deepseek-r1/deepseek-r1`，预检仅确认这些文件和 venv marker。固定版本的官方准确率脚本却把 `checkpoint_path` 写成 `deepseek-ai/DeepSeek-R1`，再按固定 revision 调用 `AutoTokenizer.from_pretrained`；这个调用不会自动使用上述模型目录。运行容器关闭网络并设置 HF offline。若最终镜像没有对应 Hub 缓存，性能测试和准确率推理完成后，评估阶段会因分词器加载失败而结束。当前文档没有填充该缓存的步骤。

   修复建议：联网准备时，在最终镜像、相同用户和相同 HF 缓存路径中，使用准确率 venv 预加载这个固定 revision 的分词器；保存缓存，并在 `--network none` 下验证相同调用。把这项检查放在长时间性能测试之前。保留官方评估源码不变时，可使用固定路径的已封装 Hub 缓存。

   官方依据：[固定评估脚本，分词器加载](https://github.com/mlcommons/inference/blob/175ea7ba24a826946458f72ec4a2221215f52802/language/deepseek-r1/eval_accuracy.py#L82)、[固定仓库名](https://github.com/mlcommons/inference/blob/175ea7ba24a826946458f72ec4a2221215f52802/language/deepseek-r1/eval_accuracy.py#L915)。此项通过源码确认，未在实际 MLPerf 镜像中执行。

2. **[P2] 被校验的配置文件不一定是官方程序实际加载的配置。**

   位置：[上下文声明](../deepseek_mlperf.py#L233)、[环境继承](../deepseek_mlperf.py#L269)。

   `check()`检查默认 `configs/B300-SXM-270GBx8/Offline/deepseek-r1.py`，但返回的 FP4、TP8、EP8 和关闭推测解码等字段是固定值。`run_phase()`继承整个环境，未处理 `CONFIG_DIR`，命令也未固定 `--config_dir`。官方程序支持通过该环境变量选择外部配置。如果准备镜像或直接运行适配器时设置了外部配置目录，实际 TP/EP、精度或其他参数可以改变，而报告仍声明默认配置并显示可比。

   修复建议：显式指定已经校验的配置目录，拒绝不允许的配置覆盖；记录并验证两个阶段实际加载的配置、生成配置、endpoint 命令和 TRT-LLM YAML。上下文应来自实际配置，而不是常量声明。

   CPU 探针确认，外部 `/synthetic/external-configs` 被传入 server，命令未指定 `--config_dir`。官方依据：[CONFIG_DIR 字段](https://github.com/mlcommons/inference_results_v6.0/blob/4d3916ac9cf474b679cdfcf492d43a0559418ad1/closed/NVIDIA/code/fields/general.py)、[配置加载入口](https://github.com/mlcommons/inference_results_v6.0/blob/4d3916ac9cf474b679cdfcf492d43a0559418ad1/closed/NVIDIA/code/main.py)。未执行改变配置后的 GPU 测试。

3. **[P2] 预处理数据只检查第一维，不能确认有效输入和官方数据口径。**

   位置：[数据校验](../deepseek_mlperf.py#L214)。

   目前两个数组的第一维等于 4388 即可通过。额外探针使用 `input_ids_padded.shape=(4388,1)`、float64 / NaN 输入以及全部为 -1 的 int64 长度，实际数据校验分支仍通过。模型、源码和 GPU 条件在这个探针中使用桩替代，没有声称这些是假模型实测。运行时记录数组 SHA256 只记录当前文件，不能证明它与官方 pickle 或固定版本预处理结果一致。错误或陈旧预处理数据可能在加载模型后才失败，也可能改变测量负载。

   修复建议：验证输入为 int32、形状 `(4388,3140)`；长度为 int32、形状 `(4388,)` 且范围有效；验证 token 范围及长度与官方原始数据对应。准备阶段生成数据清单，包含原始文件、预处理文件哈希和预处理源码版本；运行阶段核对清单。

   官方依据：[固定预处理实现](https://github.com/mlcommons/inference_results_v6.0/blob/4d3916ac9cf474b679cdfcf492d43a0559418ad1/closed/NVIDIA/code/deepseek-r1/tensorrt/preprocess_data.py)。

4. **[P2] 缺失遥测或不足 8 卡的遥测不会阻止报告 PASS。**

   位置：[报告遥测处理](../report.py#L155)。

   没有 `telemetry.csv` 时不添加复核理由；存在文件时只要求至少一条可解析记录，忽略 UUID、GPU 数量、覆盖时段和无效行。以其他证据完整的合成报告为例，删除整个遥测文件后仍为 PASS；只保留 GPU 0 的一条记录也为 PASS。DeepSeek 模式没有基础压力测试中的逐卡温度检查，因此该缺口尤其影响长时间测试的监控证据。

   修复建议：将遥测缺失和覆盖不足标为 REVIEW 或 INCOMPLETE；检查全部预期 UUID、有限数值、测试阶段覆盖以及采样中断；保留每卡温度、功耗和利用率摘要。可增加可配置监控中止条件。

5. **[P2] 准确率原始响应未验证 int32 token 格式，也未与评估摘要交叉校验。**

   位置：[响应验证](../deepseek_mlperf.py#L155)。

   当前只检查非空、偶数长度十六进制。固定官方评估默认将数据按 int32 解码，因此一个响应的十六进制长度必须是 8 的倍数。探针给全部 4388 条响应填入单字节 `01`，为修改后的合成文件更新校验和后，`load_evidence()`仍接受。现有正向夹具还用每条一个 token 的响应配上平均约 3775 token 的摘要，说明两者没有一致性校验。这是证据读取器的校验缺口，不代表官方评估流程能够产生合格的单字节响应。

   修复建议：按固定 int32 格式校验、解码 token，验证 token 数和允许的 token 值；按照官方处理规则核对输出长度与评估摘要；保存并解析 AccuracyOnly 的 detail 日志，验证版本及错误事件。新增相应故障注入测试。

   官方依据：[准确率输出默认 int32](https://github.com/mlcommons/inference/blob/175ea7ba24a826946458f72ec4a2221215f52802/language/deepseek-r1/eval_accuracy.py#L138)。

可优先考虑的优化：

- 增加独立的环境检查和小样本 smoke 模式，先验证 endpoint、分词器、准确率依赖和清理流程。smoke 结果应标为不可与 MLPerf 公测比较，完整模式继续保留完整样本和时长要求。
- 给 DeepSeek 扩展增加统一的离线资源封装与验证：数据、HF 缓存、源码依赖、准确率环境、镜像 digest 和权重都进入清单。当前基础 `bundle.py verify` 不覆盖这些外部资源，人工准备步骤较多。
- 减少重复读取约 395 GiB 权重。下载流程对已存在文件分别在容量计算、下载任务和最终验证中全量哈希；可在同一准备会话中复用已验证结果，仍保留独立完整校验入口及最终现场完整校验。
- 增加固定版本官方日志的回归夹具和 Linux 容器集成检查。现有测试主要使用简化合成日志和 mock，未覆盖官方环境实际导入、离线缓存、配置加载和脚本之间的完整衔接。
- 让报告按 profile 展示指标说明。当前 DeepSeek 报告底部仍固定说明 Qwen 独立副本、BF16 GEMM 和 SDXL；应显示本次实际运行的 DeepSeek 整机 TP/EP 测试口径。

复现材料：[CPU 探针](../tools/review_probes.py)、[探针输出](review_probe_results.json)。所有探针数据都是合成数据，模型/源码/GPU 条件按说明使用 mock，临时结果目录在运行后清理。这些材料是代码审查证据，不是 GPU 测试结果。
