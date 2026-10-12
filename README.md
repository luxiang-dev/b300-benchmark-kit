# B300 Benchmark Kit

面向 Linux 单节点 **8×NVIDIA B300** 的离线功能测试与公测性能对比工具。

**Ubuntu 22.04 / Python 3.10 和 Ubuntu 24.04 / Python 3.12 均通过 68 项测试，无跳过**，Bash、UTF-8/LF 和 CLI 检查通过。[CI 记录](https://github.com/luxiang-dev/b300-benchmark-kit/actions/runs/38196116494)。已修复离线准确率缓存、配置覆盖、数据/输出校验与遥测缺失误判等问题。**GPU 部分仍待实机验证**；CPU/模拟启动测试不代表真实 B300 测试通过，范围见 [Ubuntu 复查记录](docs/UBUNTU_REVIEW.md)。

包含 GPU/ECC/拓扑采集、BF16 计算与显存压力测试、8 卡 NCCL 通信、Qwen 与 SDXL 功能测试，以及中文 HTML/JSON 报告。

公测对比支持 Oracle 的单节点 8×B300 NCCL all-reduce 测试口径；独立的 DeepSeek-R1 扩展接入 NVIDIA 官方 MLPerf Inference v6.0 Offline 的性能和准确率流程，参考整机吞吐 **70,325.7 Tokens/s**。本地对比不代表 MLCommons 认证。

- [总使用说明：Ubuntu 准备、运行、结果与排错](docs/使用总览.md)
- [基础包详细使用说明](README_使用说明.md)
- [DeepSeek-R1 / MLPerf 专项准备与运行](README_DeepSeek_MLPerf.md)
- [Ubuntu 复查、修复与优化建议](docs/UBUNTU_REVIEW.md)
- [首次审查历史记录](docs/REVIEW.md)
- [公测来源与固定版本参数](public_baselines.json)

联网 Linux 准备机先制作基础离线包：

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
```

仓库提供源码、固定版本下载清单和准备脚本。模型权重、Docker 镜像、运行结果和本机准备状态不提交到 Git。
