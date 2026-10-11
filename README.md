# B300 Benchmark Kit

面向 Linux 单节点 **8×NVIDIA B300** 的离线功能测试与公测性能对比工具。

**当前是待实机验证的源码项目。** 已通过 56 项 CPU 测试和 Bash 语法检查，尚未构建、验证目标 Docker 镜像，也未运行 B300 GPU 测试。DeepSeek 扩展仍有待修复问题，见 [审查记录](docs/REVIEW.md)。

包含 GPU/ECC/拓扑采集、BF16 计算与显存压力测试、8 卡 NCCL 通信、Qwen 与 SDXL 功能测试，以及中文 HTML/JSON 报告。

公测对比支持 Oracle 的单节点 8×B300 NCCL all-reduce 测试口径；独立的 DeepSeek-R1 扩展接入 NVIDIA 官方 MLPerf Inference v6.0 Offline 的性能和准确率流程，参考整机吞吐 **70,325.7 Tokens/s**。本地对比不代表 MLCommons 认证。

- [完整使用说明](README_使用说明.md)
- [DeepSeek-R1 / MLPerf 专项准备与运行](README_DeepSeek_MLPerf.md)
- [已知问题、复现与优化建议](docs/REVIEW.md)
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

DeepSeek 需要另行准备完整 FP4 权重（约 395 GiB）、MLPerf 数据、固定版本官方源码依赖和独立镜像。修复已知问题并完成专项环境准备后：

```bash
export MLPERF_REPO=/data/inference_results_v6.0
export MLPERF_SCRATCH_PATH=/data/mlperf-deepseek-storage
export MLPERF_IMAGE=b300-deepseek-mlperf:6.0
bash run.sh deepseek /data/b300-results/deepseek-001
```

CPU 测试不需要下载权重或安装 GPU runtime：

```bash
python3 -m unittest discover -s tests
bash -n run.sh run_deepseek_mlperf.sh prepare_offline.sh
```

仓库提供源码、固定版本下载清单和准备脚本。模型权重、Docker 镜像、运行结果和本机准备状态不提交到 Git。
