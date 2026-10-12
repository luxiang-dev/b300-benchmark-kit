# Ubuntu 运行复查 · 2026-10-12

本次已修复首次审查中的主要问题，并加入 Ubuntu 22.04 / Python 3.10、Ubuntu 24.04 / Python 3.12 的持续验证。当前完整使用入口为 [通用验货与全流程可视化](通用验货与可视化.md)。

## 最新通用 NVIDIA / CPU 版本

受测源码为 `03fbf0d2d78a290e5484bdab614ee15f27192c77`：[实际 Ubuntu CI](https://github.com/luxiang-dev/b300-benchmark-kit/actions/runs/38229275461)。两个原始 job 日志均已核对：

| 环境 | 结果 | 说明 |
| --- | --- | --- |
| Ubuntu 24.04 / Python 3.12 | 277 tests / OK / 无跳过，44.265 秒 | job 114744444951 |
| Ubuntu 22.04 / Python 3.10 | 277 tests / OK / 无跳过，46.349 秒 | job 114744445158 |
| Windows 本地 | 277 tests / OK / 跳过 15 项 | Linux 进程/flock及符号链接权限测试不适用 |

同时通过 Bash 语法、28 个 Linux Python/脚本 UTF-8/LF、前端交互与 CLI 检查。界面实际浏览器检查覆盖 1440×1000 与 390×844：8 个项目、17 个型号选项、CPU 无 GPU 入口、H100/4 卡配置、只读预览，未发现页面异常或水平溢出。

改造内容：GPU 数量改为 1–64 张/自动探测；型号规则独立；未知型号保持 REVIEW；T4 等兼容老架构使用 FP32；仅适用设备核对 ECC/NVLink/DCGM；新增不依赖 NVIDIA/Docker 的整机流程；可视化参数随任务保存；按实际阶段显示进度，技术 PASS 不会把跳过的公测变绿。保留 8×B300 公测/原 DeepSeek 专项门禁和采购状态独立。

新增模拟测试覆盖 H100/A100/L40S/RTX/T4、16卡 UUID 对齐、型号/架构/数量矛盾、显存视角比对、消费级特性缺失、CPU 独立路径及参数校验。真实 Linux 启动器确认单卡 H100 使用一个 worker，并在模拟容器退出 42 时保留失败；H100 不能运行 B300 专属公测。**这些都是 CPU/模拟证据测试，不是实测 GPU 性能或正品认证。**

实际 GPU、默认/替代镜像构建、每种型号的 CUDA/NCCL/DCGM 兼容性和厂商完整验收仍待目标实机验证。默认 CUDA 13 镜像不保证覆盖所有历史 GPU。采购 BOM、实物来源/保修、全 RAM/全部 SSD/NIC、供电与 OEM 诊断仍需补齐；新版本不会自动收货通过。

## 先前采购验货加严版本

用户明确主要用途为大额服务器采购验货，且目前未知整机型号、具体配置和验收指标。此版本将技术 PASS 和采购验收独立：缺少书面配置、实物/保修来源及完整 OEM 验收时，采购状态固定保留 NOT_ACCEPTED / BLOCKED，不自动宣布收货通过。卖方材料清单及建议步骤见 [采购验货要求](采购验货要求.md)。

受测源码为 `8008423019c26cb7bd2606e79ce23811b4d5c14d`：[最终 CI](https://github.com/luxiang-dev/b300-benchmark-kit/actions/runs/38207992169)。两个原始 job 日志均已核对：

| 环境 | unittest | 其他检查 |
| --- | --- | --- |
| [Ubuntu 22.04 / Python 3.10](https://github.com/luxiang-dev/b300-benchmark-kit/actions/runs/38207992169/job/114675656254) | 258 项通过，无跳过 | Bash、27个Linux入口UTF-8/LF、前端、CLI 通过 |
| [Ubuntu 24.04 / Python 3.12](https://github.com/luxiang-dev/b300-benchmark-kit/actions/runs/38207992169/job/114675656144) | 258 项通过，无跳过 | 同上 |

本机 Windows 最终258项通过，13项因Linux flock/进程组或符号链接条件跳过；上述Ubuntu均实际执行。初次Ubuntu审查抓到后台遥测/sleep继承CLI锁而延迟释放的问题；已在两条Shell入口关闭监控子进程的锁FD，并在每个Linux启动器测试结束后立即断言锁可重新取得，最终CI全部通过。

此版本补齐的严重问题包括：

- 新结果绑定本轮运行、快照 SHA、UUID、PCI、CUDA架构和容量；只写PASS、少卡、重复结果或混用陈旧日志不能通过。
- 时间不足、覆盖缩水、NCCL口径不全纳入判断；显存增加独立CPU算法抽查，并用NumPy小chunk实际执行3轮测试路径，覆盖错误生产者/虚假equal/非抽查位置损坏/地址别名。
- 寿命与易失ECC、corrected新增、行重映射、NVLink/fabric、恢复动作、DRAM不可读以及非空但访问不完整的内核日志均不再被忽略。
- DCGM需可证明核心测试逐项覆盖全部GPU；未知格式/缺实体/跳过/警告不能算完整通过。
- 只读SMART/BMC前后快照重核原始字节SHA和健康字段；缓存PASS、当前故障、寿命/SEL、权限/格式缺失分别处理。RAID隐藏成员和虚拟后端仍需OEM补证。
- 派生 `observed-inventory.json`，缺证保持UNKNOWN，明确OS RAM/磁盘/接口不等于实物DIMM/SSD/NIC完整清单，不自动冒充采购BOM。
- 封存按实际ZIP字节计算SHA，证据变化拒绝导出；写盘满/挂起不阻断停止信号，取消状态不被完成边界覆盖，资源在写盘失败时仍释放。
- 网页与CLI共享锁并验证继承FD；拒绝残留结果目录，并检查systemd/cgroup内存预算。

这些是程序的CPU/模拟可靠性验证，不是买方真实8×B300的测试，也不替代厂商离线现场诊断、实物来源/保修核验、全主机RAM/成员盘/NIC/电源/冷启动验收。

## 前阶段可视化验真记录（160项）

新增中文本地工作台、PCI/CUDA/UUID/序列号交叉核验、约85% HBM 三轮地址模式写读、整机清单与有限 CPU/RAM/结果盘功能抽查、实时温度中止，以及带 SHA256 的原始证据 ZIP。支持 Ubuntu 本地浏览器与 Windows SSH 转发。硬件包不下载 Qwen/SDXL 权重，操作步骤见 [全流程说明](可视化验真操作.md)。

受测源码提交为 `4d55143be9109676d997f50c4d4ef763cd7ec932`：[本次 CI](https://github.com/luxiang-dev/b300-benchmark-kit/actions/runs/38200551379)。已读取两个 job 的原始日志，结论为：

| 环境 | unittest | 其他检查 |
| --- | --- | --- |
| [Ubuntu 22.04 / Python 3.10](https://github.com/luxiang-dev/b300-benchmark-kit/actions/runs/38200551379/job/114652770196) | 160 项通过，无跳过 | Bash、20个Linux入口UTF-8/LF、前端 smoke、CLI 通过 |
| [Ubuntu 24.04 / Python 3.12](https://github.com/luxiang-dev/b300-benchmark-kit/actions/runs/38200551379/job/114652770103) | 160 项通过，无跳过 | 同上 |

本机 Windows 同样通过160项测试，其中9项因 Linux 进程组/flock 或符号链接权限条件跳过；上面的 Ubuntu 记录实际执行了这些项目。前端另外在真实只读后端进行 1440×1000 / 390×844 浏览器检查，无页面错误和横向溢出，没有生成模拟 GPU 卡或性能。

本次重点修复：

- 摘要 PASS 不能覆盖启动器异常退出；host-only 身份采集的最终硬件/报告失败会升级退出码。
- 实时遥测核对实际 UUID 集合、索引覆盖、有限数值及范围，重复身份或丢卡立即停止。
- 显存 PASS 必须具备实际分配预算和三轮完整写读；重复读验字节不能冒充物理容量。
- 布尔值不能冒充整数 GPU rank；阶段未知状态或非零 PASS 不能通过报告。
- 本机控制接口限定 localhost/同源/token，拒绝非法参数、路径逃逸、日志符号链接及 NUL 路径。
- 取消终止本作业进程组和容器，同用户第二控制器受 flock 限制；导出时禁止取消，最终 `run-result.json` 与 ZIP 哈希一致。
- 已证明成功且没有访问警告的空内核日志窗口，和无法读取内核日志明确区分。

CPU/HTTP/启动器故障注入依旧不是 CUDA、真实 Docker GPU 镜像、NVSwitch 或 DCGM 的实机验证。整机功能抽查默认仅 256 MiB RAM / 128 MiB 临时结果文件，NIC/BMC 信息是清单，不自动证明线速或健康；证据一致不是厂家正品认证。

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

## 原阶段自动化验证记录（68项）

本机已通过 68 项 unittest（其中 4 项 Linux 专用集成测试在 Windows 上跳过），三个 Bash 脚本逐一通过语法检查，13 个 Linux 入口通过 UTF-8/LF 检查。回归探针已确认缺失/单卡遥测为 REVIEW、错误 token/dtype 被拒绝、外部 CONFIG_DIR 未继承。

GitHub 仓库已发布到正确账号 [luxiang-dev/b300-benchmark-kit](https://github.com/luxiang-dev/b300-benchmark-kit)，并于 2026-10-12 完成实际 Ubuntu runner 验证：[完整 CI 运行记录](https://github.com/luxiang-dev/b300-benchmark-kit/actions/runs/38196116494)。受测代码提交为 `35514431c8034c78aab57aea938776ced4d7db3f`。

| 环境 | unittest | Bash / UTF-8 LF | CLI smoke |
| --- | --- | --- | --- |
| [Ubuntu 22.04 / Python 3.10](https://github.com/luxiang-dev/b300-benchmark-kit/actions/runs/38196116494/job/114639012415) | 68 项全部通过，无跳过 | 通过 | 通过 |
| [Ubuntu 24.04 / Python 3.12](https://github.com/luxiang-dev/b300-benchmark-kit/actions/runs/38196116494/job/114639012269) | 68 项全部通过，无跳过 | 通过 | 通过 |

已核对两个 job 的原始日志，四个 Linux 专用启动测试均实际执行并通过。CI 对纯 Markdown 文档修改不重复运行；代码、依赖、脚本和 workflow 变化仍触发验证。

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
3. **扩展 CLI 监控与 DeepSeek smoke**：网页启动的所有 GPU 重负载和真实性 CLI 已有独立主机 watchdog；直接旧基础/DeepSeek CLI 仍需统一此能力。先短时检验 endpoint/准确率全链路，缩短结果必须标为不可与完整 MLPerf 结果比较。
4. **减少同一次准备的重复读盘**：在有可信校验记录时复用同阶段校验，现场整包/权重完整校验仍保留。
5. **真实官方日志回归和同机重复测量**：扩充成功日志兼容性样本，再根据同配置多次测量分布设置性能告警；不把任意公测百分比直接作为硬件合格线。

首次审查的历史细节保留在 [REVIEW.md](REVIEW.md)；其中行号及问题状态针对最初提交，当前状态以本页为准。
