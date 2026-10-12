# B300 Benchmark Kit

面向 Ubuntu 单节点 **8×NVIDIA B300** 的离线身份核验、整机功能抽查、稳定性测试与公测性能对比工具，提供中文可视化工作台。

**Ubuntu 22.04 / Python 3.10 和 Ubuntu 24.04 / Python 3.12 均通过 68 项测试，无跳过**，Bash、UTF-8/LF 和 CLI 检查通过。[CI 记录](https://github.com/luxiang-dev/b300-benchmark-kit/actions/runs/38196116494)。已修复离线准确率缓存、配置覆盖、数据/输出校验与遥测缺失误判等问题。**GPU 部分仍待实机验证**；CPU/模拟启动测试不代表真实 B300 测试通过，范围见 [Ubuntu 复查记录](docs/UBUNTU_REVIEW.md)。

包含 PCI/CUDA/UUID/序列号交叉核验、每卡约 85% HBM 地址模式写读、GPU/ECC/拓扑、BF16 压力、8 卡 NCCL、服务器功能抽查、Qwen/SDXL，以及可下载的 HTML/JSON/原始证据 ZIP。软件证据一致和负载通过不能替代厂商实物与采购来源确认。

公测对比支持 Oracle 的单节点 8×B300 NCCL all-reduce 测试口径；独立的 DeepSeek-R1 扩展接入 NVIDIA 官方 MLPerf Inference v6.0 Offline 的性能和准确率流程，参考整机吞吐 **70,325.7 Tokens/s**。本地对比不代表 MLCommons 认证。

- [总使用说明：Ubuntu 准备、运行、结果与排错](docs/使用总览.md)
- [8卡可视化验真：离线准备、本地/Windows 操作、报告解释](docs/可视化验真操作.md)
- [基础包详细使用说明](README_使用说明.md)
- [DeepSeek-R1 / MLPerf 专项准备与运行](README_DeepSeek_MLPerf.md)
- [Ubuntu 复查、修复与优化建议](docs/UBUNTU_REVIEW.md)
- [首次审查历史记录](docs/REVIEW.md)
- [公测来源与固定版本参数](public_baselines.json)

只核验硬件时，在 CPU 架构匹配的联网 Linux 准备机制作无需模型的硬件包：

```bash
bash prepare_offline.sh --hardware-only
python3 bundle.py verify --hardware-only
```

将完整目录复制到 Ubuntu 服务器本地 SSD 后启动：

```bash
python3 dashboard.py --port 8765 --results /data/b300-results/dashboard
```

Ubuntu 浏览器打开 `http://127.0.0.1:8765`；Windows 使用 `ssh -N -L 8765:127.0.0.1:8765 USER@SERVER` 后打开同一地址。在界面依次导入镜像、核对身份、执行整机硬件流程、查看遥测/日志并下载报告。Windows 上 `python dashboard.py --preview` 仅预览，真实 GPU 测试在 Ubuntu 运行。

命令行备用：`bash run.sh identity /data/b300-results/id-001`、`bash run.sh authenticity /data/b300-results/auth-001`。详细范围和中止方法见 [操作说明](docs/可视化验真操作.md)。

若还要 Qwen/SDXL 功能测试，准备完整基础包：

```bash
bash prepare_offline.sh
python3 bundle.py verify
```

在满足说明中驱动、Docker、Container Toolkit 等前提的空闲 8×B300 服务器执行：

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
for script in run.sh run_deepseek_mlperf.sh prepare_offline.sh; do bash -n "$script"; done
node tests/frontend_smoke.js
```

仓库提供源码、固定版本下载清单和准备脚本。模型权重、Docker 镜像、运行结果和本机准备状态不提交到 Git。
