# Linux · 8× NVIDIA B300 离线测试包

这是可携带到服务器的测试程序，包含硬件采集、BF16 计算与显存压力、8 卡 NCCL 通信、语言模型推理、图像生成、8×B300 公测对比和中文 HTML 报告。目标是功能检查、故障筛查和重复测量；厂商验收仍需对应机型的基线与官方诊断。

先阅读 [总使用说明](docs/使用总览.md)，其中汇总 Ubuntu 前提、模式选择、常见故障和最新优化建议。已验证的范围见 [Ubuntu 复查记录](docs/UBUNTU_REVIEW.md)。

**当前交付状态请先运行 `python bundle.py status` 查看。基础包须通过 `bundle.py verify` 才能离线运行；DeepSeek 扩展还需按专项说明准备独立环境。** 仅有脚本或模型权重不等于完整离线环境。

## 1. 模型和容量

| 模型 | 下载容量 | 用途 |
| --- | ---: | --- |
| Qwen3-0.6B | 1.41 GiB | 快速验证 8 张 GPU 的语言推理功能 |
| Qwen3-8B | 15.27 GiB | 8 卡独立副本推理、吞吐与响应样例 |
| SDXL base 1.0 FP16 | 6.46 GiB | 每卡生成 1024×1024 图像 |

基础模型合计约 **23.14 GiB**；完整镜像占用以构建后实际大小为准。1TB 移动硬盘足够携带基础包；基础环境准备机建议至少留 **150GB 空闲**。DeepSeek 扩展额外需要约 **395 GiB** 权重和独立镜像、数据与缓存，容量按 [专项说明](README_DeepSeek_MLPerf.md) 核算。不要使用 FAT32（单文件 4GB 上限）；可用服务器能挂载的 exFAT 或 NTFS。脚本不格式化磁盘。

模型版本和每个文件的官方哈希记录在 `models.lock.json`。下载 safetensors 权重，不下载仓库 Python 代码、不开启 `trust_remote_code`。模型卡和许可证一起携带，使用范围遵循各自许可证。

这些模型适合检查和基线测量。8B 独立副本不会占满 B300 的大显存；显存压力由专用合成测试提供。若要验证超大模型张量并行、FP8/FP4 或服务端多用户性能，需要额外准备大模型和 vLLM/TensorRT-LLM；本包不把现有结果解释为这些能力的验证。

## 2. 目标服务器先具备的条件

- Linux、8 张完整物理 B300，MIG 关闭，测试时 GPU 没有其他计算任务。
- 与服务器厂商要求一致的 NVIDIA 数据中心驱动。本包基于 CUDA 13，按 **R580 或更高** 检查；还要满足 B300 和所用镜像的具体要求。
- Docker 和 NVIDIA Container Toolkit 已安装、已配置 GPU 访问。HGX/NVSwitch 服务器的 Fabric Manager/NVLSM 按厂商配置运行。
- 主机 `python3`（3.10+）、Bash、coreutils；完整诊断需要 **B300 兼容的 DCGM 4.4+** 和正在运行的 hostengine。
- 建议将包先复制到服务器本地 SSD/NVMe，再运行；8 个进程同时从移动硬盘加载模型会明显变慢。结果也保存到本地磁盘。

**驱动、Docker、Container Toolkit、Fabric Manager、DCGM 是主机系统组件，不包含在测试镜像内。** 由于尚未确认发行版、CPU 架构和已有软件，不能用通用离线安装包替代；离线服务器若缺这些组件，需提前按发行版及厂商要求补齐。可先运行 `inventory` 采集信息。

确认 CPU：在服务器运行 `uname -m`。普通 HGX B300 通常是 x86_64，Grace/GB300 场景可能是 aarch64；镜像必须与实际 CPU 相符。

## 3. 在有网的准备机制作完整离线包

模型可以先在 Windows 上下载（Python 3.10+，只用标准库）：

```powershell
python download_models.py --plan
python download_models.py
python download_models.py --verify
```

支持断点续传，下载完校验官方源哈希；网络失败时重新运行即可。`--plan` 只读取清单，不下载权重。默认复用已锁定的版本；`--refresh` 才会重新选上游版本。

**容器需要有网的 Linux/Docker 准备机**，可以没有 GPU。把整个目录复制过去，使用与服务器相同的 CPU 架构，然后运行：

```bash
cd b300-usb-kit
bash prepare_offline.sh
python3 bundle.py verify
```

该脚本补齐模型，拉取 NVIDIA PyTorch 25.10（CUDA 13.0.2），仅解除所选应用包的版本约束，保留 NVIDIA 的 PyTorch/CUDA/NCCL 等运行时约束，安装固定版本的 Transformers/Diffusers，执行 CPU 导入检查，并编译固定源码版本的 NVIDIA nccl-tests。编译优先使用镜像中 PyTorch 已加载的 NCCL 库；若 PyTorch 静态链接，则使用镜像已有且版本完全匹配的共享库，以及对应版本头文件；源码 SHA256、编译库路径及二进制哈希留存在镜像中。随后导出完整 Docker 镜像并生成 SHA256 封装清单。若依赖约束冲突、NCCL 开发头文件缺失或镜像不支持架构，它会停止，需修正并重新验证后才能作为完成品。

默认根据准备机 `uname -m` 选择架构。跨架构构建需要准备机另外配置构建支持，不能仅改变参数就假定能运行：

```bash
TARGET_ARCH=amd64 bash prepare_offline.sh
# 或 TARGET_ARCH=arm64；优先使用与目标相同架构的准备机。
```

准备完成后目录应包含 `models/`、`images/b300-test.tar`、`images/runtime.inspect.json` 和 `bundle.sha256.json`。**本地程序的静态检查不能证明这个容器已构建成功，更不能证明 B300 实测已通过。**

## 4. 复制到 1TB 移动硬盘

完整包准备好后，把整个目录复制到移动硬盘。在 Windows 上可用脚本（以下 `E:` 是示例，换成实际盘符）：

```powershell
powershell -ExecutionPolicy Bypass -File .\copy_to_disk.ps1 -DiskRoot 'E:\' -PythonExe python
```

脚本先验证源包、检查容量，创建带时间的新目录，再验证目标包；不会格式化，也不会覆盖旧包。若只是先搬运尚未构建镜像的准备材料到有网 Linux 机，可明确加 `-AllowIncomplete`；这种副本不能直接用于离线服务器。

在 Linux 上普通复制后校验：

```bash
cp -a /path/to/b300-usb-kit /path/to/external-disk/
cd /path/to/external-disk/b300-usb-kit
python3 bundle.py verify
```

## 5. 在离线 B300 服务器运行

从硬盘复制到本地，进入目录，先采集硬件清单：

```bash
bash run.sh inventory
```

清单模式不依赖容器。保存 GPU 清单、驱动、主机、NVLink 和拓扑信息；报告显示 INCOMPLETE，表示未执行负载测试。

完整包就绪后：

```bash
bash run.sh quick    # 60秒压力、NCCL、0.6B、SDXL 功能检查
bash run.sh compare  # 环境正确性 + 8×B300 公测对应的 nccl-tests 负载
bash run.sh deepseek # 独立扩展：需先按 README_DeepSeek_MLPerf.md 准备官方环境
bash run.sh full     # 10分钟压力、NCCL、公测对比、两个 LLM、SDXL、DCGM -r 3
bash run.sh burn     # 1小时压力，然后执行 full 的其他项目
```

完整测试总时长还包含模型加载和 DCGM，不能理解为仅 10 分钟或 1 小时。运行时 `--network none`、模型加载 `local_files_only=True`，不尝试联网下载。

自定义结果目录或压力时长：

```bash
bash run.sh full /data/b300-results/test-001
BURN_SECONDS=7200 STAGE_TIMEOUT=10800 bash run.sh burn
```

默认每卡测试约 65% HBM，矩阵尺寸 8192，压力测试温度停止阈值 85°C。可用 `MEMORY_FRACTION`（最高 0.80）和 `MAX_GPU_TEMP` 调整；温度阈值应按实际机型的厂商要求设置。压力负载会长时间消耗 8 张卡的电力与散热能力，需在服务器安排的空闲测试窗口执行。

所有步骤顺序执行，失败即停止，后续未运行项标为缺失。Ctrl+C 会停止当前容器并保存已有证据。结果目录须为新目录，避免与之前的测试混合。

## 6. 查看报告

每次结果目录包含：

- `report.html`：可离线打开的中文报告与图像。
- `summary.json`、每卡 JSON：状态、耗时、吞吐、显存、模型输出、错误信息。
- `public-comparison.json`、`public-baselines.json`：公测对比结果、本次使用的固定版本参考数据。
- `nccl-public.log` / `nccl-public.json`：官方 nccl-tests 原始日志及校验后的整机结果。
- `telemetry.csv`：每 2 秒记录每卡温度、功耗、显存与 GPU 利用率。
- `before.xml` / `after.xml`：NVIDIA 硬件快照，比较不可纠正 ECC。
- `topology.txt` / `nvlink.txt`：互联清单；`*.log`：详细运行日志；`dcgm.json`：官方诊断结果。

状态解释：**FAIL**＝测试失败或发现硬件错误；**INCOMPLETE**＝必测项未运行/跳过/缺少 GPU 结果；**REVIEW**＝缺乏日志、存在历史 ECC 或其他需要人工复核的证据；**PASS**＝本次已执行项目通过。读取内核日志可能需要当前账户已有权限；没有有效日志时不会宣称 Xid 已检查通过。

负载运行的退出码：0 为 PASS，1 为失败（具体测试或启动错误也可能返回其他非零码），2 为 REVIEW / INCOMPLETE。清单模式本身返回 0，报告仍显示 INCOMPLETE。报告程序已做断点续传、损坏模型、遗漏单卡结果、ECC 增量、Xid、DCGM 跳过等 CPU 故障注入验证；实际 GPU 功能和性能需在 B300 服务器上验证。

语言模型测量是 8 个独立副本，吞吐包含预填充和解码；不提供首 Token 延迟或张量并行结果。NCCL 性能使用 MAX all-reduce，另做 SUM 数据正确性检查；总线带宽折算系数为 `2×(N−1)/N`，不能直接当成 NVLink 链路带宽或证明 NVLink 路由。BF16 GEMM 指标排除显存检查时间，是合成负载指标。

对照同型号服务器、同驱动/镜像、同功耗设置的基线，关注每卡差异、降频、持续功耗与错误。仅凭“程序跑完”和任意 TFLOPS 门槛不能认定硬件验收合格。

## 7. 与 8×B300 公测数据对比

内置参考位于 `public_baselines.json`，携带源链接、固定 Git 提交、核实日期、GPU/节点数、软件、负载和单位，运行时完全离线。当前选择的是已核实的公开记录，不宣称为最新榜单。

### 可测量对比：单机 8 卡 NCCL all-reduce

[Oracle OCI 公开记录（固定版本）](https://github.com/oracle-quickstart/oci-gpu-quickstarts/blob/b4570535c0e8e6fbc01745bc50bfd9064787cb0a/nvidia/B300/README-B300.md#all-reduce---single-node) 使用 `BM.GPU.B300.8`，FP32 SUM，单进程、单线程、8 卡，512 KiB 至 8 GiB、每次翻倍，1 次预热、20 次测量、校验开启、CUDA Graph 关闭。采用其 **8 GiB 消息行**：

| 布局 | 延迟 (us) | 算法带宽 (GB/s) | 总线带宽 (GB/s) |
| --- | ---: | ---: | ---: |
| 非原地 | 18006 | 477.05 | 834.84 |
| 原地 | 17990 | 477.49 | 835.61 |

`GB/s` 是十进制带宽；8 GiB 是 8589934592 字节。总线带宽为 `algbw × 2×(8−1)/8`，已经是 NCCL 的折算指标，不能再乘 8，也不等于物理 NVLink 链路速率。报告不拿跨消息尺寸的平均值或两节点 16 卡结果作基线。

运行 `bash run.sh compare`，或执行 `full` / `burn`，会调用与公测参数一致的 NVIDIA `all_reduce_perf`，并在报告中列出本次值、公测值、本次/公测百分比及差距。延迟越低越好，带宽越高越好。`quick` 保留短时功能检查，报告的公测项显示未运行。

公测 NCCL 是 2.28.9 / CUDA 13.1，本包镜像使用自身已有版本。公测没有披露 nccl-tests 版本；本包固定为 v2.17.6。报告会显示这些差异，并提示结合 CPU、拓扑、固件、功耗、散热及主机硬件快照核对。**“口径一致”仅指测量负载相符，不表示服务器全部配置相同。**

解析器要求日志完整、全部 8 张 B300、同一主机/通信组、每个尺寸齐全、两种布局均为零错误、有限正数指标及正确折算系数。测试参数不一致时显示 `CONFIG_MISMATCH`，不生成百分比；日志缺失、校验失败或哈希不一致时不会显示成功对比。性能低于公测只展示差距，不自动判定硬件故障；功能状态与性能参考分别保留。

公测测试默认最多 1800 秒，可用 `NCCL_PUBLIC_TIMEOUT` 修改。8 GiB 消息测试会使用较多显存，仍需空闲 GPU。升级脚本后，原封装清单不再有效：需重新执行 `bash prepare_offline.sh` 构建、导出和封装镜像，再 `python3 bundle.py verify`。

已有官方 nccl-tests 日志也可在 CPU 机器离线导入，使用新目录：

```bash
python3 public_benchmark.py import-nccl --log /data/all_reduce_perf.log --out /data/imported-b300
python3 public_benchmark.py compare --out /data/imported-b300
```

导入日志必须包含 nccl-tests 版本/库版本和全部 GPU 设备行。命令生成 `public-comparison.html` 和 JSON，仅表示该日志的性能对比，不生成硬件验收通过结论。要重建已有完整测试报告，执行 `python3 report.py /data/b300-results/test-001`。可在 `compare` 命令使用 `--baselines /path/public_baselines.json` 选择自行核实的参考文件；参考必须保留同样的溯源和口径字段。

### 推理公测参考

[MLCommons 官方 MLPerf Inference v6.0 汇总（固定版本）](https://github.com/mlcommons/inference_results_v6.0/blob/ab33b2144a9f822f69d24f78569f993d195ee7e5/summary_results.json) 的 NVIDIA `6.0-0073`：单节点 8×B300，Closed / Offline，FP4，整机输出 Tokens/s：

| 模型 | 公开整机吞吐 (Tokens/s) |
| --- | ---: |
| DeepSeek-R1 | 70325.7 |
| Llama2-70B-99 | 112954.0 |
| Llama3.1-8B | 165432.0 |

DeepSeek-R1 已提供独立的 `deepseek` 扩展模式，调用 NVIDIA 官方 MLPerf v6.0 Offline 性能和准确率流程，通过原始日志校验后计算整机吞吐对比。准备、模型下载和运行步骤见 [DeepSeek 测试说明](README_DeepSeek_MLPerf.md)。当前包仍未准备这些约 395 GiB 的 FP4 权重、MLPerf 数据集与独立 runtime，不能直接离线执行此扩展。

常规 Qwen3 BF16 / Transformers 独立副本与 MLPerf 的模型、精度、数据集、输入输出长度和调度不同，仍标记 `REFERENCE_ONLY`。Llama 项目前仅保留公测参考。SDXL 与合成 BF16 GEMM 也不套用上述数字。

## 8. 官方研究依据（核实于 2026-10-11）

- [NVIDIA PyTorch 25.10 / CUDA 13.0.2](https://docs.nvidia.com/deeplearning/frameworks/pytorch-release-notes/rel-25-10.html)
- [CUDA 驱动兼容性](https://docs.nvidia.com/deploy/cuda-compatibility/minor-version-compatibility.html)
- [DCGM 版本说明：4.4 增加 B300 诊断支持](https://docs.nvidia.com/datacenter/dcgm/latest/release-notes/changelog.html)
- [DCGM 诊断及运行等级](https://docs.nvidia.com/datacenter/dcgm/latest/user-guide/dcgm-diagnostics.html)
- [NVIDIA Container Toolkit 安装与 Docker 配置](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)
- [Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B)、[Qwen3-8B](https://huggingface.co/Qwen/Qwen3-8B)、[SDXL base](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0)
- [vLLM 对 B300 的 CUDA 13 建议（如后续扩展服务测试）](https://docs.vllm.ai/en/latest/getting_started/installation/gpu/)
- [NVIDIA nccl-tests 参数说明](https://github.com/NVIDIA/nccl-tests/tree/da0b547b1b9c6e3b1d4c15578087874522ae3761)
- [NVIDIA nccl-tests 带宽口径](https://github.com/NVIDIA/nccl-tests/blob/da0b547b1b9c6e3b1d4c15578087874522ae3761/doc/PERFORMANCE.md)
