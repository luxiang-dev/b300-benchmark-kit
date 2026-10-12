# Ubuntu 运行复查 · 2026-10-12

本次已修复首次审查中的主要问题，并加入 Ubuntu 22.04 / Python 3.10、Ubuntu 24.04 / Python 3.12 的持续验证。完整使用入口为 [总使用说明](使用总览.md)。

## 已修复

| 问题 | 修复与验证 |
| --- | --- |
| 官方准确率按 HF 仓库名加载 tokenizer，原有本地模型目录不等于 Hub 缓存 | 新增 `prepare_deepseek_accuracy.py`，从固定哈希的本地元数据创建对应 Hub snapshot；预检核对缓存哈希，并在准确率 venv 中离线实际加载 |
| 官方 LiveCodeBench 还会读取数据集，离线准确率可能迟至推理完成才失败 | 新增联网 warm 入口；预检实际加载 PRM 数学检查器与官方 LiveCodeBench，固定 HF_HOME/Hub/datasets cache 并禁用联网 |
| CONFIG_DIR / Make 环境可覆盖被检查的配置 | 两阶段显式使用 `--config_dir=configs --config_id=default` 和固定 accuracy/power；清除继承的 Make 覆盖变量，记录 launch.json |
| 数据只检查首维 4388 | 要求 int32、完整 shape、合法输入长度和词表范围；逐条与官方原始 pickle 的 token/长度核对，检查 EOS=1 padding，并记录原始数据 hash |
| 准确率原始响应接受任意偶数 hex | 要求小端 int32 token、合法 token 范围、唯一整数 QSL 索引，检查平均输出长度与 accuracy.txt 一致；新增准确率 detail 的版本/错误验证 |
| 遥测缺失或只覆盖一张卡仍可 PASS | 检查 UUID、索引、有限数值、八卡覆盖、采集起止窗口及超过 30 秒的间断；按卡汇总，缺失需 REVIEW，过热 FAIL，采集进程提前退出导致失败 |
| GB300 可因名称包含 B300 而通过基础检查 | 使用独立单词匹配 B300，增加 Linux 启动器误型号回归 |
| DeepSeek 镜像未检查 CPU 架构、未检查驱动大版本 | 与基础模式一致，检查 host/image 架构及 R580+；具体驱动补丁仍需满足厂商要求 |
| harness 超时可能遗留 make 的子进程 | 为 harness 创建独立进程组，超时/退出后清理；端点服务依旧按阶段重新启动并清理 |
| DeepSeek 报告底部仍描述 Qwen 独立副本 | 按模式显示指标解释；明确整机吞吐、本地复现和认证边界 |
| Windows 复制会携带 .git、CI、运行结果 | 从基础携带包复制中排除这些目录；DeepSeek 外部资源保留独立 Linux 搬运路径 |

新的轻量环境预检在读取约 395 GiB 权重之前进行，避免环境明显不完整时先花时间扫描全部权重；最终完整模型 hash 校验仍保留。

## 自动化验证范围

本机已通过 68 项 unittest（其中 4 项 Linux 专用集成测试在 Windows 上跳过），三个 Bash 脚本逐一通过语法检查，13 个 Linux 入口通过 UTF-8/LF 检查。回归探针已确认缺失/单卡遥测为 REVIEW、错误 token/dtype 被拒绝、外部 CONFIG_DIR 未继承。

待执行 GitHub workflow：[Ubuntu validation](https://github.com/luxiang20041020-eng/b300-benchmark-kit/actions/workflows/ubuntu.yml)。目标账号已按用户更正为 luxiang20041020-eng；目前连接与本机凭据仍显示旧账号，推送还遇到 GitHub TLS 连接失败。**尚未在 Ubuntu runner 执行**，实际运行链接在正确账号授权及推送完成后补充。

验证包含 CPU 故障注入、数据一致性、token 格式/长度、离线缓存准备，以及真实 Linux 下的 inventory、容器阶段错误码传播、失败后报告/清理、GB300 拒绝和 DeepSeek Docker 挂载参数。Linux 启动测试明确使用模拟 nvidia-smi / Docker，临时 seal 和数据仅为测试夹具，不代表真实硬件或镜像。

所有 CPU 测试不需要联网权重或 GPU runtime；使用 `requirements-dev.txt` 中固定 NumPy/Pandas。在 Windows 上 Linux 进程测试会跳过；必须看 Ubuntu workflow 才能确认这些路径。

本机 WSL 虽登记了 Ubuntu，但其 ext4.vhdx 缺失，不能启动；未修复用户 WSL 环境。Git Bash 用于独立 Shell 语法检查。

## 尚未证实的运行层

- 没有真实 8×B300，未验证 CUDA/NCCL 内核、DCGM、Qwen/SDXL 或完整 DeepSeek 的运行结果。
- 未在本机构建 NGC 基础镜像和官方 DeepSeek 镜像；其包约束、开发头文件、NGC 访问与硬件初始化仍需实际准备机确认。
- 没有携带完整 DeepSeek 数据/模型/venv；新离线 evaluator 检查必须在最终官方镜像执行通过，不能凭 CPU 测试宣称通过。
- 原始 pickle 需由官方 R2 Downloader 获取；当前内容核对证明原始数据与预处理一致，未新增独立的官方数据源哈希锁。
- 此工具不自动执行 TEST06、正式提交检查或 MLCommons 认证。

## 后续优化优先级

1. **实机闭环**：在 Ubuntu 24.04、空闲 8×B300 上依次保存 quick / compare / full / deepseek 成功或失败的原始证据，补齐最终软件与镜像 ID。
2. **统一 DeepSeek 资源清单**：将外部源码、venv、数据、HF 缓存、镜像 tar 的 hash 和位置封装起来。当前基础 seal 仅覆盖基础包。
3. **独立 smoke 与温度 watchdog**：先短时检验 endpoint/准确率全链路；缩短结果标为不可与完整 MLPerf 结果比较。DeepSeek 的温度目前在报告阶段判定，尚未实时终止服务。
4. **减少同一次准备的重复读盘**：在有可信校验记录时复用同阶段校验，现场整包/权重完整校验仍保留。
5. **真实官方日志回归和同机重复测量**：扩充成功日志兼容性样本，再根据同配置多次测量分布设置性能告警；不把任意公测百分比直接作为硬件合格线。

首次审查的历史细节保留在 [REVIEW.md](REVIEW.md)；其中行号及问题状态针对最初提交，当前状态以本页为准。
