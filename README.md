# Server Acceptance Kit

面向 Ubuntu 的通用采购验货辅助工具：支持单节点 **1–64 张 NVIDIA GPU**，以及**无需 GPU 的整机检查**。提供中文全流程可视化：目标配置 → 设备识别 → 项目选择 → 参数设置 → 阶段/温度/日志 → 停止与清理 → 清单、报告及证据下载。

**主要用于采购验货辅助，技术 PASS 不等于整机验收完成。** 新报告独立显示采购状态与待补证项目；配置清单、厂商来源/保修和完整整机诊断缺失时不自动收货通过。买方暂不清楚配置时，先导出实测清单再让卖方核对，见 [采购验货要求与卖方材料清单](docs/采购验货要求.md)。

通用验货、公网映射及离线资源工具均有回归测试。**Ubuntu 22.04 / Python 3.10、Ubuntu 24.04 / Python 3.12 各通过 307 项测试，无跳过**，Bash/编码/前端/CLI 检查通过；见 [本次 CI](https://github.com/luxiang-dev/b300-benchmark-kit/actions/runs/38334657299)。**不同 GPU 和运行时仍需实机验证**；CPU/模拟启动测试不代表真实 GPU 验货或目标机断网运行通过，验证范围见 [Ubuntu 复查记录](docs/UBUNTU_REVIEW.md)。

包含 PCI/CUDA/UUID/序列号交叉核验、每卡默认约 85% 显存地址模式写读、适用的 ECC/拓扑、按架构使用 BF16/FP32 的压力、多卡 NCCL、服务器功能抽查、可选 Qwen/SDXL，以及 HTML/JSON/原始证据 ZIP。已知架构规则与通用一致性检查分开；未知型号保留 REVIEW。消费级卡没有 ECC/NVLink 不直接判假，也不能记录为已测通过。

公测对比支持 Oracle 的单节点 8×B300 NCCL all-reduce 测试口径；独立的 DeepSeek-R1 扩展接入 NVIDIA 官方 MLPerf Inference v6.0 Offline 的性能和准确率流程，参考整机吞吐 **70,325.7 Tokens/s**。本地对比不代表 MLCommons 认证。

- [通用验货与全流程可视化：准备、本地/Windows操作、参数、结果](docs/通用验货与可视化.md)
- [局域网/公网映射：监听地址、代理与可选口令](docs/公网映射访问.md)
- [通用离线资源总清单：基础包、DeepSeek、主机依赖及断网验练](docs/离线资源总清单.md)
- [原 B300 专项：Ubuntu 准备、运行、结果与排错](docs/使用总览.md)
- [原 8×B300 可视化验真说明](docs/可视化验真操作.md)
- [采购验货门禁、当前遗漏与厂商材料清单](docs/采购验货要求.md)
- [基础包详细使用说明](README_使用说明.md)
- [DeepSeek-R1 / MLPerf 专项准备与运行](README_DeepSeek_MLPerf.md)
- [Ubuntu 复查、修复与优化建议](docs/UBUNTU_REVIEW.md)
- [首次审查历史记录](docs/REVIEW.md)
- [公测来源与固定版本参数](public_baselines.json)

无 GPU 或仅查整机部件时，只需 Python 3 和源码；无需准备镜像。GPU 负载测试在 CPU 架构匹配的联网 Linux 准备机制作无需模型的硬件包：

```bash
bash prepare_offline.sh --hardware-only
python3 bundle.py verify --hardware-only
```

将完整目录复制到 Ubuntu 服务器本地 SSD 后启动：

```bash
python3 dashboard.py --host 0.0.0.0 --port 8765 --results /data/b300-results/dashboard
```

Ubuntu 浏览器打开 `http://127.0.0.1:8765`；Windows 使用 `ssh -N -L 8765:127.0.0.1:8765 USER@SERVER` 后打开同一地址。在界面填写预期型号/卡数和测试参数，再选择“整机检查（无需 GPU）”或 GPU 流程，监控并导出报告。未知采购型号选自动识别、卡数填 0；此时不核实是否少配。Windows `python dashboard.py --preview` 为只读预览。

默认监听所有 IPv4 网卡，也可直接访问 `http://服务器IP:8765` 或你自己的公网映射 URL，无需 SSH 隧道。映射端负责 HTTPS/认证；应用提供可选 `--access-token-file`。仅本机使用指定 `--host 127.0.0.1`，代理改写 Host 时指定 `--public-origin`，见 [公网映射说明](docs/公网映射访问.md)。

命令行备用：`bash run.sh server /data/server-results/cpu-001`、`bash run.sh identity /data/server-results/id-001`、`bash run.sh authenticity /data/server-results/gpu-001`。详细范围和中止方法见 [通用操作说明](docs/通用验货与可视化.md)。镜像默认 CUDA 13；老 GPU 须选择兼容的 `BASE_IMAGE` 重新构建，不能保证默认镜像覆盖所有历史型号。旧镜像标签/文件名、仓库地址保留兼容，不限制型号。

若还要 Qwen/SDXL 功能测试，准备完整基础包：

```bash
bash prepare_offline.sh
python3 bundle.py verify
```

`quick`/`full`/`burn` 按实际卡数运行预置模型，仍有架构/显存要求。`compare` 和原 DeepSeek 专项限定 8×B300，不能套用到其他机器：

```bash
bash run.sh quick /data/b300-results/quick-001
bash run.sh compare /data/b300-results/nccl-001
```

DeepSeek 需要另行准备完整 FP4 权重（约 395 GiB）、MLPerf 数据、固定版本官方源码依赖、离线准确率缓存和独立镜像。完成专项环境准备后：

```bash
export MLPERF_REPO=/data/inference_results_v6.0
export MLPERF_SCRATCH_PATH=/data/mlperf-deepseek-storage
export MLPERF_IMAGE=b300-deepseek-mlperf:6.0
bash run.sh deepseek /data/b300-results/deepseek-001
```

CPU 测试不需要下载权重或安装 GPU runtime；新增数据格式测试使用 NumPy/Pandas，建议在独立 venv 中运行：

```bash
python3 -m venv .venv-test
. .venv-test/bin/activate
python -m pip install -r requirements-dev.txt
python -m unittest discover -s tests
for script in run.sh run_deepseek_mlperf.sh prepare_offline.sh verify_offline.sh; do bash -n "$script"; done
node tests/frontend_smoke.js
```

仓库提供源码、固定版本下载清单和准备脚本。模型权重、Docker 镜像、运行结果和本机准备状态不提交到 Git。

离线准备在页面选择“离线资源核对”，可选无 GPU、硬件、基础模型或基础加 DeepSeek 四档。核对显示具体缺项和哈希状态，不下载、不安装；仅文件存在不会宣布离线就绪。CLI 对应 `python3 offline_resources.py --scope all --verify-hashes --out /data/results/offline-inventory`。准备完成后在目标 Ubuntu 执行 `bash verify_offline.sh full`（全部资源用 `all`），再运行真实硬件/模型流程。外部 DeepSeek、数据/缓存及主机安装材料可用 `offline_manifest.py` 制作、验证完整性清单。详见 [离线总清单](docs/离线资源总清单.md)。
