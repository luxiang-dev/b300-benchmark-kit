# DeepSeek-R1 · MLPerf Inference v6.0 Offline · 8×B300

本扩展接入 NVIDIA 官方 LoadGen + TensorRT-LLM endpoint 测试流程，使用完整 DeepSeek-R1 FP4 权重、单节点 8 张物理 B300、TP=8 / EP=8，以及完整 4388 条 MLPerf 样本。执行 `PerformanceOnly` 后重新启动 endpoint，再执行 `AccuracyOnly`。这不是 Qwen 独立副本测试，也不是蒸馏模型测试。

公测参考为 NVIDIA `6.0-0073` 的 8×B300 Closed / Offline 结果：**70325.7 Tokens/s（整机输出吞吐）**。[固定版本官方结果](https://github.com/mlcommons/inference_results_v6.0/blob/ab33b2144a9f822f69d24f78569f993d195ee7e5/summary_results.json)。对比只提供本地复现参考；当前入口未自动执行审计 TEST06、提交检查和认证流程。

## 当前交付状态

已提供运行适配器、独立 Docker 启动脚本、固定版本模型哈希清单、断点续传下载器及报告校验。**此电脑没有准备 DeepSeek 权重、数据集、官方源码依赖或 MLPerf 镜像，也没有运行任何 B300 测试。** `prepare_offline.sh` 仍负责原有 Qwen/SDXL/NCCL 包；它不会自动准备本扩展。

FP4 权重及其配置约 **394.56 GiB**，另有分词器、原始/预处理数据、源码、镜像、准确率输出和缓存空间。下载器使用预量化权重，只下载原始模型的分词器/配置元数据；若自行重新量化，还需要单独准备原始完整权重及额外空间。复制前按实际准备目录核算容量。

## 1. 在联网 Linux 主机准备官方环境

官方运行说明：[NVIDIA 主 README](https://github.com/mlcommons/inference_results_v6.0/blob/4d3916ac9cf474b679cdfcf492d43a0559418ad1/closed/NVIDIA/README.md)、[DeepSeek 专项 README](https://github.com/mlcommons/inference_results_v6.0/blob/4d3916ac9cf474b679cdfcf492d43a0559418ad1/closed/NVIDIA/code/deepseek-r1/tensorrt/README.md)。以下路径是假设，按存储实际位置设置：

```bash
export KIT=/data/b300-usb-kit
export MLPERF_REPO=/data/inference_results_v6.0
export MLPERF_SCRATCH_PATH=/data/mlperf-deepseek-storage
git clone https://github.com/mlcommons/inference_results_v6.0.git "$MLPERF_REPO"
git -C "$MLPERF_REPO" checkout 4d3916ac9cf474b679cdfcf492d43a0559418ad1
mkdir -p "$MLPERF_REPO/closed/NVIDIA/3rdparty"
git clone https://github.com/mlcommons/inference.git "$MLPERF_REPO/closed/NVIDIA/3rdparty/mlc-inference"
git -C "$MLPERF_REPO/closed/NVIDIA/3rdparty/mlc-inference" checkout 175ea7ba24a826946458f72ec4a2221215f52802
git -C "$MLPERF_REPO/closed/NVIDIA/3rdparty/mlc-inference" submodule update --init --recursive
```

NVIDIA 导出不包含 `3rdparty/`。还需按主 README 准备 `3rdparty/trtllm` 及运行依赖，实际 TRT-LLM 使用匹配的 release 镜像内已安装版本。不要用其他轮次的规则替换这里的 MLCommons 固定提交；当前 `master` 已包含 v6.1，官方加载器会选择最新版本。

主 README 给出的 x86 release 镜像示例：

```bash
export RELEASE_IMAGE=nvcr.io/nvidia/mlperf/mlperf-inference:tensorrt_llm_release-feat-1.2-mlpinf-b5ddff4_mlperf-main-f538816_jan28_x86
docker pull "$RELEASE_IMAGE"
```

在联网阶段根据目标服务器架构和官方说明完成环境准备，并固定最终镜像 ID。此处未验证该镜像能在当前主机运行；NGC 访问、驱动兼容性与实际 GPU 初始化需在准备机/目标机确认。

## 2. 模型与数据

在 NVIDIA 环境内先设置 `MLPERF_SCRATCH_PATH` 并执行 `make link_dirs ENV=release`，使 `build/models`、`build/data`、`build/preprocessed_data` 指向 scratch。之后可在挂载相同路径的准备机执行：

```bash
python3 "$KIT/deepseek_assets.py" download --workspace "$MLPERF_REPO/closed/NVIDIA" --workers 2
python3 "$KIT/deepseek_assets.py" verify --workspace "$MLPERF_REPO/closed/NVIDIA"
```

下载器固定 `nvidia/DeepSeek-R1-FP4@4bedb8a695a119b1a38d16a675c4665e58708aea`，完整 80 个 safetensors 分片；原始分词器/配置固定 `deepseek-ai/DeepSeek-R1@56d4cbbb4d29f4355bab4b9a39ccb717a14ad5ad`。文件清单保留源站 SHA256 / Git blob 哈希。不会自动切换到 FP4-v2；官方称 v2 可能更快，当前公测结果未明确披露具体权重版本，报告会保留这一差异。

按[固定 MLCommons 数据说明](https://github.com/mlcommons/inference/blob/175ea7ba24a826946458f72ec4a2221215f52802/language/deepseek-r1/README.md)使用官方 R2 Downloader 和元数据地址 `https://inference.mlcommons-storage.org/metadata/deepseek-r1-datasets-fp8-eval.uri`，将以下文件放到 `build/data/deepseek-r1/`：

- `mlperf_deepseek_r1_dataset_4388_fp8_eval.pkl`
- `mlperf_deepseek_r1_calibration_dataset_500_fp8_eval.pkl`

在官方容器中运行 `BENCHMARKS=deepseek-r1 make preprocess_data ENV=release`，确认输出 `build/preprocessed_data/deepseek-r1/input_ids_padded.npy` 和 `input_lens.npy`，均覆盖 4388 条样本，并保留校准文件。不要根据文件名中的 `fp8_eval` 把推理精度改成 FP8：它是数据版本名称，当前推理用 FP4。

## 3. 准确率环境与离线镜像

容器需要同时能访问完整仓库 `/submission` 和 NVIDIA 工作目录 `/work`，因为官方准确率依赖包含指向 `/work/3rdparty/...` 的 editable 包。准备与测试时均挂载 scratch 的原绝对路径，以保留 `build/` 链接：

```bash
docker run -it --name b300-deepseek-prepare --gpus all --shm-size=32g \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  -e MLPERF_SCRATCH_PATH="$MLPERF_SCRATCH_PATH" \
  -v "$MLPERF_REPO:/submission" -v "$MLPERF_REPO/closed/NVIDIA:/work" \
  -v "$MLPERF_SCRATCH_PATH:$MLPERF_SCRATCH_PATH" -v "$KIT:/kit:ro" \
  -w /work --entrypoint /bin/bash "$RELEASE_IMAGE"
```

在容器内完成官方运行依赖安装、数据预处理和 LoadGen 编译安装；LoadGen 源码必须来自上述固定 MLCommons 提交（可在该仓库 `loadgen/` 执行 `python3 -m pip install .`）。按官方准确率初始化逻辑移除 `prm800k/setup.py` 的 `import numpy` 行，然后预先创建准确率环境：

```bash
python3 - <<'PY'
from pathlib import Path
from code.common.mlcommons.venv_utils import ensure_venv_ready
setup = Path('3rdparty/mlc-inference/language/deepseek-r1/submodules/prm800k/setup.py')
setup.write_text(''.join(line for line in setup.read_text().splitlines(keepends=True) if 'import numpy' not in line))
ensure_venv_ready(Path('/work/.dsr1-acc-venv'), Path('code/deepseek-r1/tensorrt/requirements.accuracy.txt'))
PY
python3 /kit/prepare_deepseek_accuracy.py --workspace /submission/closed/NVIDIA --warm-evaluator
python3 /kit/deepseek_mlperf.py check --workspace /submission/closed/NVIDIA
```

新增准备入口会将固定版本本地 tokenizer 元数据复制到 `build/hf-cache/hub` 的对应 Hub snapshot，校验源哈希；`--warm-evaluator` 联网加载官方 PRM 数学检查器及 LiveCodeBench 数据缓存。不要省略 LiveCodeBench 缓存，官方准确率流程会使用它。运行时固定使用 `build/hf-cache`，关闭 HF Hub、Transformers 和 datasets 联网。

`check` 先在准确率 venv 中以离线模式实际加载固定 tokenizer、PRM 数学检查器及 LiveCodeBench，再检查固定源码、数据 int32 格式与原始 pickle 的逐样本对应、运行依赖和 8 张 B300，最后逐文件读取并校验约 395 GiB 权重。必须在目标 GPU 环境通过预检。源码、模型、数据、缓存与 `.dsr1-acc-venv` 保留在挂载目录；系统级安装保留到最终镜像。退出准备容器后：

```bash
docker commit b300-deepseek-prepare b300-deepseek-mlperf:6.0
docker image inspect b300-deepseek-mlperf:6.0 > deepseek-runtime.inspect.json
docker save -o b300-deepseek-mlperf.tar b300-deepseek-mlperf:6.0
```

外置盘需同时携带：本测试包、完整官方仓库（包括 `.git`、`3rdparty` 和 `.dsr1-acc-venv`）、scratch 模型/数据目录（包括 hf-cache）、镜像 tar 与 inspect JSON。原包 `bundle.py verify` 不校验这些外部扩展资源。Linux 源码/venv 依赖符号链接与执行权限；推荐直接使用 ext4/XFS，若经 exFAT 搬运则先在 Linux 将完整仓库和 scratch 打包为 tar，再到目标 Linux 文件系统解包。复制后保持 scratch 的绝对路径；Python 环境必须使用同一容器内 `/work` 路径。源码和 scratch 路径不要有空格或冒号。

在最终镜像验证完全断网预检（按实际目录设置变量）：

```bash
docker run --rm --pull never --network none --gpus all --shm-size=32g \
  --entrypoint /bin/bash \
  -e MLPERF_SCRATCH_PATH="$MLPERF_SCRATCH_PATH" \
  -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 -e HF_DATASETS_OFFLINE=1 \
  -v "$MLPERF_REPO:/submission" -v "$MLPERF_REPO/closed/NVIDIA:/work" \
  -v "$MLPERF_SCRATCH_PATH:$MLPERF_SCRATCH_PATH" -v "$KIT:/kit:ro" \
  -w /submission/closed/NVIDIA b300-deepseek-mlperf:6.0 \
  -c 'exec python3 /kit/deepseek_mlperf.py check --workspace /submission/closed/NVIDIA'
```

## 4. 在 8×B300 Linux 服务器运行

```bash
docker load -i /data/b300-deepseek-mlperf.tar
export MLPERF_REPO=/data/inference_results_v6.0
export MLPERF_SCRATCH_PATH=/data/mlperf-deepseek-storage
export MLPERF_IMAGE=b300-deepseek-mlperf:6.0
bash /data/b300-usb-kit/run.sh deepseek /data/b300-results/deepseek-001
```

使用新的输出目录、空闲的 8 张物理 B300、MIG 关闭且 ECC 开启。启动器解析本地镜像为 ID，禁止拉取镜像并关闭容器外部网络；两个阶段分别启动新的推理服务，默认服务启动上限 2 小时、每个阶段上限 12 小时。600000 ms 是最低配置要求：105312 条性能样本按公测 18.7489 samples/s 估算约 94 分钟，另加模型加载、哈希校验和准确率评估时间。此模式独立于 `full` / `burn`，不会在常规验机中自动加入数小时的完整模型测试。

生成 `report.html`、`summary.json`、`public-comparison.json`、`deepseek-mlperf.json`、两个阶段的 endpoint/harness/LoadGen 日志、准确率输出、镜像信息、ECC/内核/拓扑和遥测。

## 5. 结果判定

只有固定版本、完整模型 FP4、单节点 8 GPU、TP8/EP8、Offline，且两个阶段成功、原始证据 SHA256 一致时才比较吞吐。性能摘要必须 `VALID`、满足时长与查询约束、配置最低时长 600000 ms、至少 105312 条 issued samples，以及完整 4388 条 QSL。

准确率必须覆盖 4388 个唯一整数 QSL 索引，响应为合法词表范围的小端 int32 token，原始平均 token 长度与 accuracy.txt 一致，并通过官方准确率执行流程；性能和准确率 detail 都必须记录 v6.0 LoadGen 且无错误事件。适配器另外复核 `exact_match >= 81.3582 × 0.99 = 80.544618`，平均输出长度位于 `3886.2274 × [0.9, 1.1]`。[v6.0 官方判定常量](https://github.com/mlcommons/inference/blob/175ea7ba24a826946458f72ec4a2221215f52802/tools/submission/submission_checker/constants.py)。日志缺失/损坏/缩短运行/不同模型或精度均不会计算百分比。

报告显示整机 Tokens/s、公测 70325.7、本次/公测及差距，不把单卡值乘 8。公测性能偏差不自动判为硬件故障；软件、CPU、功耗、拓扑和权重版本差异需结合报告核对。未运行时保留 `REFERENCE_ONLY`，不会用 Qwen 吞吐代替本项目。

## 离线资源补充入口

基础包和本专项全部离线依赖、运输与空间规划见 [通用离线资源总清单](docs/离线资源总清单.md)。`offline_resources.py --scope all --verify-hashes --out /data/results/offline-inventory` 可核对现有资源，不下载。外部源码/scratch/镜像使用 `offline_manifest.py` 封装并在目标机验证；基础 `bundle.py` 不覆盖这些外部资源。

准备完成并设置 `MLPERF_REPO`、`MLPERF_SCRATCH_PATH`、`MLPERF_IMAGE` 后，目标 Ubuntu 执行 `bash verify_offline.sh deepseek` 做无外网容器中的专项 check；随后仍需实际执行性能及准确率测试。可额外设置 `MLPERF_IMAGE_ARCHIVE`（最终本地 tar）与 `MLPERF_IMAGE_INSPECT`（最终 inspect JSON），供核对/本地导入使用；启动可视化控制服务前设置这些变量。
