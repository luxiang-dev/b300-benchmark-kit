"use strict";

// No external libraries, network resources, or estimated measurements.
const $ = (id) => document.getElementById(id);
const dom = {};
document.querySelectorAll("[id]").forEach((node) => { dom[node.id] = node; });
const profileInfo = {
  identity: {title:"身份交叉检查", subtitle:"硬件身份 · 架构与驱动", category:"IDENTITY CHECK", icon:"◈", action:"开始身份检查", requires_gpus:0, requires_docker:false, requires_image:false, scope:["型号、UUID、PCI 与 VBIOS","满足条件时检查 CUDA 能力","显存与硬件身份一致性","逐卡异常与缺失证据"], description:"先采集主机身份，识别型号、架构和容量之间的不一致。仅在完整 8 卡与本地镜像可用时补充 CUDA 检查；缺失项目保留为未完成或需复核。", boundary:"结论是软件证据的一致性评估。驱动提供的名称、UUID 或序列号本身不能代替厂商认证；还应核对实物、采购记录与供应商信息。"},
  authenticity: {title:"真实性负载", subtitle:"服务器抽查 · 显存 · 计算 · 互联", category:"HARDWARE VALIDATION", icon:"⌘", action:"开始真实性负载", requires_gpus:8, requires_docker:true, requires_image:true, scope:["逐卡身份与 CUDA 能力检查","服务器清单与 CPU / RAM / 结果盘抽查","显存容量与数据正确性压力","实际计算与 NCCL 互联负载","ECC、Xid、温度与原始日志"], description:"对整台 8×B300 服务器运行真实硬件负载，补充 CPU、RAM 与结果盘的小样本正确性检查，并采集存储、NIC、NUMA 与 BMC 清单。此项目无需大模型权重。", boundary:"服务器抽查默认只覆盖 256 MiB RAM 与 128 MiB 结果盘文件。全 RAM、全盘、NIC 远端线速和 OEM 人工验收仍需另行完成；本次通过仅适用于实际已执行的范围。"},
  quick: {title:"快速功能", subtitle:"短压力 · Qwen · SDXL", category:"FUNCTIONAL SMOKE TEST", icon:"ϟ", action:"开始快速功能", requires_gpus:8, requires_docker:true, requires_image:true, requires_bundle:true, scope:["短时计算与显存压力","PyTorch NCCL 通信","Qwen3-0.6B 每卡独立推理","SDXL 图像生成"], description:"在完整基础离线包上运行短时功能检查，确认计算、通信与模型推理可以工作，再安排完整功能测试。", boundary:"此处 Qwen 为每张卡独立副本，模型与精度不同，吞吐不能直接与 DeepSeek-R1 MLPerf 整机公测比较。"},
  compare: {title:"NCCL 公测对比", subtitle:"8 卡整机 · 相同测试条件", category:"PUBLIC REFERENCE", icon:"⇄", action:"开始 NCCL 对比", requires_gpus:8, requires_docker:true, requires_image:true, requires_bundle:true, scope:["8 张物理 B300 与环境检查","固定 nccl-tests 原始输出","8 GiB All-Reduce 对应条件","带宽、正确性与公测比例"], description:"运行与公开 8×B300 参考相对应的 NCCL 测试，保留原始输出与参数，比较整机通信能力。", boundary:"仅在卡数、数据大小、操作与数据类型等条件相符时比较。参考成绩是定位差异的依据，达不到某个比例本身不能证明是假卡。"},
  full: {title:"完整功能测试", subtitle:"压力 · 公测 · 模型 · DCGM", category:"FULL FUNCTIONAL TEST", icon:"▦", action:"开始完整功能测试", requires_gpus:8, requires_docker:true, requires_image:true, requires_bundle:true, scope:["600 秒计算与显存压力","NCCL 通信与公测对比","两种 Qwen 及 SDXL 推理","DCGM、ECC 与内核错误复核"], description:"运行基础包的全部功能项目，覆盖硬件压力、通信、两种文本模型、图像模型与诊断，汇总为一份技术报告。", boundary:"full/burn 需要可用的 DCGM。缺失诊断项目时报告保留为未完成。此项目不自动运行独立的 DeepSeek MLPerf 扩展，也不完成整机采购验收。"},
  burn: {title:"持续压力", subtitle:"可选时长 · 完整功能测试后续", category:"SUSTAINED STRESS", icon:"◷", action:"开始持续压力", requires_gpus:8, requires_docker:true, requires_image:true, requires_bundle:true, scope:["指定时长计算与显存压力","逐卡温度与功耗遥测","完整通信及模型功能测试","运行前后 ECC 与错误检查"], description:"延长压力阶段，观察整机供电、散热和稳定性；压力结束后继续执行完整功能测试的其他项目。", boundary:"所选时长仅指压力阶段，总运行时间还包含模型加载、通信和诊断。先通过短时检查，再运行较长负载。"},
  deepseek: {title:"DeepSeek MLPerf", subtitle:"R1 FP4 · v6.0 Offline", category:"MLPERF INFERENCE", icon:"◎", action:"开始 DeepSeek 测试", requires_gpus:8, requires_docker:true, requires_deepseek:true, scope:["官方固定源码、模型与数据校验","DeepSeek-R1 FP4 · TP8 / EP8","Offline 性能与准确率两阶段","8 卡整机 Tokens/s 公测对比"], description:"通过独立准备的官方运行环境，执行 DeepSeek-R1 的 MLPerf Inference v6.0 Offline 性能与准确率流程。完整负载耗时较长。", boundary:"必须提前完成独立资源和准确率缓存准备。参考为 8 卡整机 70,325.7 Tokens/s；本地测试结果不是 MLCommons 认证提交。"},
};
const stageLabels = {identity:"身份与架构检查", server:"服务器清单与 CPU / RAM / 结果盘抽查", "server-health-before":"压力前 SMART / BMC 健康", "server-health-after":"压力后 SMART / BMC 健康", "host-identity":"主机身份采集", "identity-host":"主机身份采集", "identity-cuda":"CUDA 身份检查", "identity-evaluation":"身份一致性评估", capacity:"显存容量与正确性", memory:"显存检查", environment:"GPU 运行环境", stress:"计算与显存压力", nccl:"PyTorch NCCL 通信", "nccl-public":"NCCL 公测对比", dcgm:"DCGM 诊断", preflight:"环境与资源预检", "llm-Qwen3-0.6B":"Qwen3-0.6B 推理", "llm-Qwen3-8B":"Qwen3-8B 推理", "image-SDXL-base-1.0":"SDXL 图像生成", "deepseek-r1-mlperf":"DeepSeek-R1 MLPerf", telemetry:"遥测采集", launcher:"测试启动与清理", report:"汇总报告", integrity:"资源完整性校验"};
const statusLabels = {PASS:"通过",CONSISTENT:"信号一致",FAIL:"失败",REVIEW:"需复核",INCOMPLETE:"未完成",RUNNING:"运行中",PENDING:"待执行",STOPPED:"已停止",UNKNOWN:"待复核",SKIP:"已跳过"};
let state = null;
let csrfToken = null;
let selectedProfile = "identity";
let selectedJobId = null;
let selectionIsManual = false;
let connected = false;
let pendingMutation = false;
let refreshing = false;
let nextPoll = null;
let toastTimer = null;

function node(tag, className, text) {
  const result = document.createElement(tag);
  if (className) result.className = className;
  if (text !== undefined && text !== null) result.textContent = String(text);
  return result;
}
function replace(container, children) { container.replaceChildren(...children); }
function finite(value) { return typeof value === "number" && Number.isFinite(value); }
function numeric(value) { return (value !== null && value !== undefined && value !== "" && Number.isFinite(Number(value))) ? Number(value) : null; }
function displayNumber(value, digits = 1) { const number = numeric(value); return number === null ? "—" : number.toLocaleString("zh-CN", {maximumFractionDigits:digits}); }
function available(value) {
  if (value && typeof value === "object") return value.available === true || value.ok === true || value.present === true;
  return value === true || (typeof value === "string" && value !== "" && value !== "missing" && value !== "unavailable");
}
function rawStatus(value) {
  const text = String(value || "UNKNOWN").toUpperCase();
  if (["SUCCESS","SUCCEEDED","PASSED"].includes(text)) return "PASS";
  if (["FAILURE","FAILED","ERROR"].includes(text)) return "FAIL";
  if (["ACTIVE","STARTING","STOPPING","EXPORTING"].includes(text)) return "RUNNING";
  if (["QUEUED","WAITING"].includes(text)) return "PENDING";
  if (["CANCELLED","CANCELED","INTERRUPTED"].includes(text)) return "STOPPED";
  if (["SKIPPED"].includes(text)) return "SKIP";
  return statusLabels[text] ? text : "UNKNOWN";
}
function jobStatus(job) {
  if (!job) return "UNKNOWN";
  const status = rawStatus(job.status);
  if (["RUNNING","PENDING","STOPPED","FAIL"].includes(status)) return status;
  const summaryStatus = rawStatus(job.summary?.overall || job.summary?.status || job.identity_summary?.status || job.identity_summary?.verdict);
  return summaryStatus !== "UNKNOWN" ? summaryStatus : status;
}
function badge(status, translated = false) {
  const normalized = rawStatus(status);
  const classes = {PASS:"pass",CONSISTENT:"pass",FAIL:"fail",REVIEW:"review",INCOMPLETE:"incomplete",RUNNING:"running",PENDING:"pending",STOPPED:"stopped",UNKNOWN:"review",SKIP:"incomplete"};
  return node("span", `status-badge status-${classes[normalized]}`, translated ? statusLabels[normalized] : normalized);
}
function dateLabel(value, full = false) {
  if (value === null || value === undefined || value === "") return "—";
  const timestamp = typeof value === "number" ? (value < 1e12 ? value * 1000 : value) : value;
  const date = new Date(timestamp);
  if (Number.isNaN(date.getTime())) return String(value);
  return date.toLocaleString("zh-CN", full ? {hour12:false} : {month:"2-digit",day:"2-digit",hour:"2-digit",minute:"2-digit",hour12:false});
}
function jobs() { return Array.isArray(state?.jobs) ? state.jobs : []; }
function activeJob() {
  if (state?.active_job && typeof state.active_job === "object") return state.active_job;
  return jobs().find((job) => job.id === state?.active_job || rawStatus(job.status) === "RUNNING") || null;
}
function profile(id) {
  const base = id === "import-image" ? {title:"导入离线镜像",requires_gpus:0,requires_docker:true} : profileInfo[id];
  const result = {...base, ...(state?.profiles || []).find((item) => item.id === id), id};
  // Older controllers may still call the full profile "完整验收". Its scope is a
  // technical test, so keep the reviewed wording while honoring eligibility.
  if (["quick","full","burn"].includes(id)) {
    for (const key of ["title","subtitle","category","action","description","boundary"]) result[key] = base[key];
  }
  return result;
}
function readiness() { return state?.readiness || {}; }
const acceptanceStatement = "测试通过不等于采购验收完成。请勿仅依据 PASS 或身份信号一致判定整机验收通过或支付尾款。";
const defaultAcceptanceChecks = [
  {id:"bom",title:"采购 BOM 与每台实物、BMC 清单逐项比对",status:"PENDING"},
  {id:"vendor",title:"厂家核验服务器及部件序列号、保修与来源",status:"PENDING"},
  {id:"ram",title:"全 RAM 容量与稳定性测试",status:"PENDING"},
  {id:"storage",title:"全部 SSD 健康、容量与读写验证",status:"PENDING"},
  {id:"network",title:"所有 NIC 端口与远端链路线速验证",status:"PENDING"},
  {id:"power",title:"供电、冗余与冷启动后的整机复测",status:"PENDING"},
];
function acceptanceFor(job) {
  const value = job?.summary?.acceptance;
  const blockers = Array.isArray(value?.blockers) ? value.blockers.filter((item) => typeof item === "string" && item).slice(0,30) : [];
  if (jobStatus(job) === "FAIL" && !blockers.length) blockers.push("本次技术检查存在失败，需排查并保留原始证据。");
  const blocked = value?.status === "BLOCKED" || blockers.length > 0;
  const supplied = Array.isArray(value?.pending_checks) ? value.pending_checks.filter((item) => item && typeof item.title === "string" && item.title).slice(0,30).map((item) => ({...item,status:["FAIL","REVIEW"].includes(item.status) ? item.status : "PENDING"})) : [];
  const checks = supplied.length ? supplied : defaultAcceptanceChecks;
  return {status:blocked ? "BLOCKED" : "NOT_ACCEPTED", blockers, checks, statement:acceptanceStatement};
}
function renderAcceptance() {
  const job = jobs().find((entry) => entry.id === selectedJobId) || activeJob() || jobs()[0] || null;
  const acceptance = acceptanceFor(job);
  const blocked = acceptance.status === "BLOCKED";
  dom.acceptancePanel.classList.toggle("acceptance-blocked",blocked);
  dom.acceptanceTitle.textContent = blocked ? "采购验收存在阻断项" : "采购验收尚未完成";
  dom.acceptanceStatus.textContent = acceptance.status;
  dom.acceptanceStatement.textContent = acceptance.statement;
  dom.acceptanceContext.textContent = job ? `当前记录：${profile(job.profile).title || job.profile} · ${job.id}。下面是尚需补齐的验货证据，不会由技术 PASS 自动变为通过。` : "未提供采购 BOM、实物及厂商核验记录，也未完成全部整机测试。";
  dom.acceptanceBlockers.hidden = !acceptance.blockers.length;
  replace(dom.acceptanceBlockers,acceptance.blockers.map((detail) => node("p",null,detail)));
  replace(dom.acceptanceChecks,acceptance.checks.map((check) => {
    const row = node("div","acceptance-check");const label = badge(check.status,true);if (check.status === "PENDING") label.textContent = "待补证";row.append(label,node("span",null,check.title));return row;
  }));
}
function gate(id, ignoreBusy = false) {
  const info = profile(id);
  if (!connected || !state) return {ok:false, reason:"尚未连接测试主机。"};
  if (state.demo || state.read_only) return {ok:false, reason:"演示 / 只读模式不能启动实际测试。"};
  if (pendingMutation && !ignoreBusy) return {ok:false, reason:"正在提交操作，请稍候。"};
  if ((state.busy || activeJob()) && !ignoreBusy) return {ok:false, reason:"有任务运行中；请等待完成或先停止任务。"};
  if (info.available === false || info.enabled === false) return {ok:false, reason:info.unavailable_reason || info.reason || "此测试项目当前不可运行。"};
  const r = readiness();
  const tools = r.tools || {};
  const gpus = Array.isArray(r.gpus) ? r.gpus : [];
  if (!available(tools.python3)) return {ok:false, reason:"主机未检测到可用的 Python 3。"};
  if (id !== "import-image" && !available(tools.nvidia_smi)) return {ok:false, reason:"主机未检测到可用的 nvidia-smi，请检查驱动。"};
  const required = numeric(info.requires_gpus);
  if (required !== null && required > 0 && gpus.length !== required) return {ok:false, reason:`此项目需要 ${required} 张物理 GPU，当前识别到 ${gpus.length} 张。`};
  if (!["import-image","identity"].includes(id) && gpus.some((gpu) => !/\bB300\b/i.test(String(gpu.name || gpu.product_name || "")))) return {ok:false, reason:"存在型号名称不是 B300 的设备，请先核对主机。"};
  if (info.requires_docker && !available(tools.docker)) return {ok:false, reason:"Docker 不可用或无访问权限。"};
  if (info.requires_image && !available(r.image)) return {ok:false, reason:"尚未准备本地基础测试镜像。请完成离线准备或导入镜像。"};
  if (info.requires_bundle && r.bundle?.sealed !== true) return {ok:false, reason:"基础离线包尚未封存，请完成准备并校验资源。"};
  if (info.requires_deepseek) {
    const ds = r.deepseek || {};
    const prepared = ds.configured === true || ds.prepared === true || ds.available === true || (available(ds.repo) && available(ds.image) && available(ds.scratch));
    if (!prepared) return {ok:false, reason:"DeepSeek 独立源码、镜像或 scratch 路径尚未准备。请按专项说明设置环境变量。"};
  }
  return {ok:true, reason:"运行条件已识别；任务启动后还会执行严格预检。"};
}

async function api(path, options = {}) {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), 15000);
  try {
    const response = await fetch(path, {credentials:"same-origin", cache:"no-store", ...options, signal:controller.signal});
    let body;
    try { body = await response.json(); } catch { body = {}; }
    if (!response.ok) {
      if (response.status === 403) csrfToken = null;
      throw new Error(body.error || body.message || `请求失败（HTTP ${response.status}）`);
    }
    return body;
  } catch (error) {
    if (error.name === "AbortError") throw new Error("主机响应超时，请确认服务和 SSH 隧道仍在运行。");
    throw error;
  } finally { clearTimeout(timer); }
}
async function session() {
  const result = await api("/api/session");
  csrfToken = result.csrf_token;
  if (typeof csrfToken !== "string" || !csrfToken) throw new Error("服务未返回操作会话，无法启动测试。");
}
function showAction(message, error = false) {
  dom.actionMessage.textContent = message;
  dom.actionMessage.classList.toggle("alert-error", error);
  dom.actionMessage.hidden = false;
}
function toast(message) {
  dom.toast.textContent = message;
  dom.toast.hidden = false;
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => { dom.toast.hidden = true; }, 3200);
}

function renderReadiness() {
  const r = readiness();
  const tools = r.tools || {};
  const gpus = Array.isArray(r.gpus) ? r.gpus : [];
  const rows = [
    {name:"Python 3",detail:"主机启动与报告生成",ok:available(tools.python3)},
    {name:"NVIDIA 驱动",detail:"nvidia-smi 与 GPU 读取",ok:available(tools.nvidia_smi)},
    {name:"Docker",detail:"容器运行与访问权限",ok:available(tools.docker)},
    {name:"本地测试镜像",detail:r.image?.architecture ? `架构：${r.image.architecture}` : "身份检查与硬件负载",ok:available(r.image)},
    {name:"基础离线包",detail:"Qwen / SDXL / 公测资源",ok:r.bundle?.sealed === true},
    {name:"8 张物理 B300",detail:`${gpus.length} 张设备已读取`,ok:gpus.length === 8 && gpus.every((gpu) => /\bB300\b/i.test(String(gpu.name || "")))},
  ];
  replace(dom.readinessList, rows.map((row) => {
    const item = node("div", "readiness-row");
    item.append(node("span", `readiness-indicator ${connected ? (row.ok ? "ok" : "missing") : "wait"}`, connected ? (row.ok ? "✓" : "!") : "·"));
    const label = node("div");
    label.append(node("div","readiness-row-name",row.name),node("div","readiness-row-detail",row.detail));
    item.append(label,node("span","readiness-row-status",connected ? (row.ok ? "已检测" : "待准备") : "待读取"));
    return item;
  }));
  const passed = rows.filter((row) => row.ok).length;
  dom.readinessTag.textContent = connected ? `${passed} / ${rows.length} 已检测` : "等待连接";
  const importGate = gate("import-image");
  dom.importImageControl.hidden = available(r.image);
  dom.importImageButton.disabled = !importGate.ok;
  dom.importImageReason.textContent = !importGate.ok ? importGate.reason : "校验准备包后导入 images/b300-test.tar；不从网络拉取。";
  const warningValues = Array.isArray(r.warnings) ? r.warnings : [];
  dom.warningsDetails.hidden = warningValues.length === 0;
  dom.warningCount.textContent = `${warningValues.length} 项环境提醒`;
  replace(dom.warningsList, warningValues.map((warning) => node("li",null,typeof warning === "object" ? warning.message || warning.description || JSON.stringify(warning) : warning)));
}

function renderHardware() {
  const r = readiness();
  const gpus = Array.isArray(r.gpus) ? r.gpus : [];
  dom.sideGpuCount.textContent = state ? String(gpus.length) : "—";
  dom.gpuCount.textContent = state ? String(gpus.length) : "—";
  const memories = gpus.map((gpu) => numeric(gpu.memory_mib));
  dom.totalMemory.textContent = gpus.length && memories.every((value) => value !== null) ? displayNumber(memories.reduce((sum,value) => sum + value,0) / 1024,1) : "—";
  dom.gpuCountNote.textContent = gpus.length === 8 ? "8 卡整机 · 身份仍需交叉检查" : gpus.length ? "当前卡数与 8 卡公测条件不同" : connected ? "未读取到 GPU，请检查驱动与权限" : "等待 nvidia-smi 数据";
  dom.hardwareTag.textContent = gpus.length ? `${gpus.length} 张设备` : "无设备数据";
  if (!gpus.length) {
    const empty = node("div","empty-state");
    empty.append(node("span","empty-symbol","◈"),node("strong",null,connected ? "尚未识别到 GPU" : "等待设备信息"),node("p",null,connected ? "请确认 NVIDIA 驱动可用，且服务在目标服务器运行。" : "连接测试主机后显示型号、UUID 和显存。"));
    replace(dom.gpuGrid,[empty]);
  } else {
    replace(dom.gpuGrid,gpus.map((gpu,index) => {
      const card = node("div",`gpu-card${/\bB300\b/i.test(String(gpu.name || "")) ? "" : " unexpected"}`);
      const top = node("div","gpu-card-top");
      top.append(node("span","gpu-index",`GPU ${gpu.index ?? index}`),node("span","gpu-memory",`${displayNumber(numeric(gpu.memory_mib) === null ? null : Number(gpu.memory_mib) / 1024,1)} GiB`));
      card.append(top,node("div","gpu-name",gpu.name || gpu.product_name || "未提供型号"),node("div","gpu-uuid",gpu.uuid || "未提供 UUID"));
      if (gpu.ecc || gpu.mig) {
        const modes = node("div","gpu-modes");
        if (gpu.ecc) modes.append(node("span",gpu.ecc === "Enabled" ? "mode-ok" : "mode-review",`ECC ${gpu.ecc === "Enabled" ? "开启" : gpu.ecc === "Disabled" ? "关闭" : gpu.ecc}`));
        if (gpu.mig) modes.append(node("span",gpu.mig === "Disabled" || gpu.mig === "N/A" ? "mode-ok" : "mode-review",`MIG ${gpu.mig === "Disabled" ? "关闭" : gpu.mig === "Enabled" ? "开启" : gpu.mig}`));
        card.append(modes);
      }
      return card;
    }));
  }
  const platform = state?.platform;
  dom.hostName.textContent = typeof platform === "object" ? platform.hostname || platform.node || "测试主机" : platform || "等待连接";
  dom.hostDetail.textContent = typeof platform === "object" ? [platform.system || platform.os || "Ubuntu",platform.release || platform.version,platform.machine || platform.architecture].filter(Boolean).join(" · ") : platform ? "主机执行 · 本地访问" : "Ubuntu 本地服务";
}

function renderProfiles() {
  const ids = Object.keys(profileInfo);
  replace(dom.profileList,ids.map((id) => {
    const info = profile(id);
    const button = node("button",`profile-option${selectedProfile === id ? " selected" : ""}`);
    button.type = "button";
    button.setAttribute("aria-pressed",String(selectedProfile === id));
    button.append(node("span","profile-icon",info.icon));
    const label = node("span");
    label.append(node("span","profile-title",info.title),node("span","profile-subtitle",info.subtitle));
    button.append(label,node("span","profile-arrow","›"));
    button.addEventListener("click",() => { selectedProfile = id; renderProfiles(); renderSelected(); });
    return button;
  }));
}
function renderSelected() {
  const info = profile(selectedProfile);
  dom.selectedCategory.textContent = info.category;
  dom.selectedTitle.textContent = info.title;
  dom.selectedDescription.textContent = selectedProfile === "authenticity" && !/抽查|CPU|RAM/i.test(info.description || "") ? `${info.description} 同时采集服务器清单，并抽查 CPU、RAM 与结果盘。` : info.description;
  dom.selectedRequirements.textContent = selectedProfile === "identity" ? "主机身份 · CUDA 按条件补充" : info.requires_deepseek ? "8 卡 · 独立 MLPerf 资源" : info.requires_bundle ? "8 卡 · 完整基础包" : `${info.requires_gpus || 8} 卡 · 本地镜像`;
  replace(dom.selectedScope,info.scope.map((text) => {
    const item = node("div","scope-item");item.append(node("span",null,"○"),node("div",null,text));return item;
  }));
  dom.profileBoundary.textContent = info.boundary;
  dom.durationControl.hidden = selectedProfile !== "burn";
  const eligibility = gate(selectedProfile);
  dom.startButton.disabled = !eligibility.ok;
  dom.startButtonLabel.textContent = pendingMutation ? "正在提交…" : info.action;
  dom.launchReason.textContent = eligibility.reason;
  dom.launchReason.classList.toggle("blocked",!eligibility.ok && connected);
}

function renderWorkflow() {
  const items = [
    {name:"运行条件",profiles:[],ready:true},
    {name:"身份检查",profiles:["identity","authenticity"]},
    {name:"真实负载",profiles:["authenticity","full","burn"]},
    {name:"互联公测",profiles:["compare","full","burn"]},
    {name:"模型推理",profiles:["quick","full","burn","deepseek"]},
    {name:"报告复核",profiles:[],evidence:true},
  ];
  replace(dom.workflowSteps,items.map((item,index) => {
    let status = "PENDING";
    if (item.ready) status = connected && gate("identity",true).ok ? "PASS" : "PENDING";
    else if (item.evidence) {
      const job = jobs().find((entry) => entry.id === selectedJobId);
      if (job && Array.isArray(job.artifacts) && job.artifacts.length && rawStatus(job.status) !== "RUNNING") status = "REVIEW";
    } else {
      const job = jobs().find((entry) => item.profiles.includes(entry.profile));
      if (job) status = jobStatus(job);
    }
    const cls = ["PASS","CONSISTENT"].includes(status) ? " done" : status === "RUNNING" ? " running" : status === "FAIL" ? " failed" : "";
    const step = node("div",`workflow-step${cls}`);
    const symbol = ["PASS","CONSISTENT"].includes(status) ? "✓" : status === "RUNNING" ? "●" : String(index + 1).padStart(2,"0");
    step.append(node("span","step-number",symbol));
    const label = node("div");
    label.append(node("div","step-title",item.name),node("div","step-state",item.evidence && status === "REVIEW" ? "报告可复核" : status === "PENDING" ? "待检查" : statusLabels[status]));
    step.append(label);
    return step;
  }));
}

function selectAutomaticJob() {
  const current = jobs().find((job) => job.id === selectedJobId);
  if (selectionIsManual && current) return;
  selectedJobId = activeJob()?.id || jobs()[0]?.id || null;
}
function renderJobHistory() {
  const entries = jobs();
  dom.historyCount.textContent = String(entries.length);
  dom.runCount.textContent = String(entries.length);
  const active = activeJob();
  const latest = active || entries[0];
  const status = latest ? jobStatus(latest) : "PENDING";
  dom.runStatus.textContent = active ? "运行中" : latest ? statusLabels[status] || "待复核" : "待检查";
  dom.runStatus.style.color = status === "FAIL" ? "var(--red)" : ["REVIEW","INCOMPLETE","UNKNOWN"].includes(status) ? "var(--amber)" : status === "RUNNING" ? "var(--blue)" : "var(--accent)";
  dom.runStatusNote.textContent = latest ? `${profile(latest.profile).title || latest.profile} · ${dateLabel(latest.started_at)}` : "先完成主机准备与身份检查";
  dom.runCountNote.textContent = active ? "一个任务正在执行" : entries.length ? "每次运行独立保存证据" : "尚未开始测试";
  if (!entries.length) { replace(dom.jobList,[node("div","history-empty","开始测试后，运行记录会显示在这里。")]); return; }
  replace(dom.jobList,entries.map((job) => {
    const button = node("button",`job-option${job.id === selectedJobId ? " selected" : ""}`);
    button.type = "button";
    button.setAttribute("aria-pressed",String(job.id === selectedJobId));
    const top = node("div","job-option-title");
    top.append(node("span",null,profile(job.profile).title || job.profile),badge(jobStatus(job),true));
    button.append(top,node("div","job-option-date",dateLabel(job.started_at)),node("div","job-id",job.id));
    button.addEventListener("click",() => { selectedJobId = job.id; selectionIsManual = true; renderJobHistory(); renderJobDetail(); renderWorkflow(); renderAcceptance(); });
    return button;
  }));
}
function stagesFor(job) {
  const latest = new Map();
  for (const stage of Array.isArray(job?.stages) ? job.stages : []) latest.set(stage.name,stage);
  return Array.from(latest.values());
}
function renderStages(job) {
  const stages = stagesFor(job);
  if (!stages.length) { replace(dom.stagesList,[node("div","stages-empty",job ? "尚无已记录阶段；以实际日志为准。" : "开始测试后显示实际执行阶段。")]); return; }
  replace(dom.stagesList,stages.map((stage) => {
    const status = rawStatus(stage.status);
    const row = node("div","stage-row");
    const symbol = node("span","stage-symbol",status === "PASS" ? "✓" : status === "FAIL" ? "×" : status === "RUNNING" ? "●" : "·");
    symbol.style.color = status === "PASS" ? "var(--accent)" : status === "FAIL" ? "var(--red)" : "var(--subtle)";
    row.append(symbol,node("span","stage-name",stageLabels[stage.name] || stage.name || "未命名阶段"));
    if (stage.exit_code !== null && stage.exit_code !== undefined) row.append(node("span","stage-code",`exit ${stage.exit_code}`));
    row.append(badge(status,true));
    return row;
  }));
}

function sparkline(samples) {
  const values = samples.map((sample) => numeric(typeof sample === "number" ? sample : sample.temperature_c ?? sample.temperature)).filter((value) => value !== null);
  if (values.length < 2) return null;
  const namespace = "http://www.w3.org/2000/svg";
  const svg = document.createElementNS(namespace,"svg");
  svg.setAttribute("viewBox","0 0 120 28");svg.setAttribute("class","sparkline");svg.setAttribute("preserveAspectRatio","none");
  svg.setAttribute("role","img");svg.setAttribute("aria-label",`${values.length} 个实际温度采样的变化趋势`);
  const min = Math.min(...values), max = Math.max(...values), range = max - min;
  const polyline = document.createElementNS(namespace,"polyline");
  polyline.setAttribute("points",values.map((value,index) => `${index * 120 / (values.length - 1)},${range === 0 ? 14 : 25 - (value - min) / range * 22}`).join(" "));
  svg.append(polyline);return svg;
}
function renderTelemetry(job) {
  const live = job?.telemetry && Object.keys(job.telemetry.per_gpu || {}).length > 0;
  const telemetry = live ? job.telemetry : job?.summary?.telemetry || {};
  const perGpu = telemetry.per_gpu || {};
  const entries = Array.isArray(perGpu) ? perGpu.map((item,index) => [item.uuid || String(index),item]) : Object.entries(perGpu);
  dom.telemetryPanel.hidden = entries.length === 0;
  dom.telemetryTitle.textContent = live ? "实时遥测" : "遥测证据";
  replace(dom.telemetryGrid,entries.map(([uuid,data]) => {
    const match = (readiness().gpus || []).find((gpu) => gpu.uuid === uuid);
    const card = node("div","telemetry-card");
    card.append(node("div","telemetry-name",match ? `GPU ${match.index} · ${uuid.slice(-8)}` : uuid));
    const temp = numeric(data.latest_temperature_c ?? data.latest?.temperature_c);
    const limit = numeric(job?.summary?.max_gpu_temp) ?? 85;
    const temperature = node("div",`telemetry-temperature${temp !== null && temp >= limit ? " hot" : ""}`,displayNumber(temp,0));
    temperature.append(node("span","unit"," °C"));card.append(temperature);
    const values = node("div","telemetry-values");
    values.append(node("div",null,`峰值 ${displayNumber(data.peak_temperature_c,0)} °C`),node("div",null,`功耗 ${displayNumber(data.latest_power_w ?? data.latest?.power_w,0)} W`));
    const samples = Array.isArray(data.samples) ? data.samples : Array.isArray(telemetry.samples) ? telemetry.samples.filter((sample) => sample.uuid === uuid).slice(-240) : [];
    const chart = sparkline(samples);
    if (chart) card.append(chart);
    const count = numeric(Array.isArray(data.samples) ? null : data.samples) ?? numeric(data.sample_count) ?? samples.length;
    if (count !== null && count !== 0) values.append(node("div",null,`${live ? "窗口 " : ""}${displayNumber(count,0)} 次采样`));
    card.append(values);return card;
  }));
}

function findingsFrom(job) {
  const findings = [];
  for (const source of [job?.identity_summary,job?.summary]) {
    if (!source || typeof source !== "object") continue;
    if (source.message || source.conclusion) findings.push({title:"评估说明",message:source.message || source.conclusion,status:source.status || source.verdict});
    for (const key of ["findings","failures","review","incomplete","warnings"]) {
      const values = source[key];
      if (!Array.isArray(values)) continue;
      for (const item of values) findings.push(typeof item === "string" ? {message:item,status:key === "failures" ? "FAIL" : key === "incomplete" ? "INCOMPLETE" : "REVIEW"} : {title:item.title || item.name || item.check || item.code,message:item.message || item.description || item.reason || item.detail,status:item.status || item.severity || (key === "failures" ? "FAIL" : "REVIEW")});
    }
    if (Array.isArray(source.checks)) for (const check of source.checks) findings.push({title:check.title || check.name || check.check,message:check.message || check.description || check.reason || check.detail,status:check.status});
  }
  if (job?.identity_summary?.vendor_confirmation === "NOT_PERFORMED") findings.push({title:"厂商核验尚未执行",message:"请将采集的序列号、部件编号与实物铭牌、BMC 和采购记录对照，并联系厂商或供应商确认。软件信号一致不等于厂商认证。",status:"REVIEW"});
  return findings.filter((finding) => finding.message || finding.title).slice(0,30);
}
function renderIdentity(job) {
  const entries = job?.identity_summary?.per_gpu;
  dom.identityPanel.hidden = !Array.isArray(entries) || entries.length === 0;
  if (!Array.isArray(entries)) { dom.identityTable.replaceChildren();return; }
  const table = node("table","identity-table");
  const head = node("thead");const header = node("tr");
  ["GPU / 型号","PCI 身份","CUDA 架构","显存观察值","序列号 / VBIOS","一致性"].forEach((text) => { const cell = node("th",null,text);cell.scope="col";header.append(cell); });
  head.append(header);table.append(head);
  const body = node("tbody");
  entries.forEach((item) => {
    const row = node("tr");
    const gpu = node("td");gpu.append(node("strong",null,`GPU ${item.host_index ?? "—"}`),node("small",null,item.name || "型号未提供"));
    const pci = node("td");pci.append(node("code",null,item.pci_bus_id || "—"),node("small",null,`${item.pci_ids?.vendor || "—"}:${item.pci_ids?.device || "—"}`));
    const architecture = node("td");architecture.append(node("strong",null,Array.isArray(item.capability) ? item.capability.join(".") : item.capability || "—"),node("small",null,"Compute Capability"));
    const memory = node("td");memory.append(node("span",null,`CUDA ${displayNumber(numeric(item.cuda_memory_bytes) === null ? null : Number(item.cuda_memory_bytes) / (2 ** 30),1)} GiB`),node("small",null,`NVML ${displayNumber(numeric(item.memory_total_bytes) === null ? null : Number(item.memory_total_bytes) / (2 ** 30),1)} GiB`));
    const serial = node("td");serial.append(node("code",null,item.serial || "—"),node("small",null,item.vbios_version || "VBIOS 未提供"));
    const status = node("td");status.append(badge(item.status || "REVIEW",true));
    row.append(gpu,pci,architecture,memory,serial,status);body.append(row);
  });
  table.append(body);replace(dom.identityTable,[table]);
}
function renderFindings(job) {
  const findings = findingsFrom(job);
  dom.findingsPanel.hidden = findings.length === 0;
  if (!findings.length) { dom.findingsPanel.replaceChildren(); return; }
  const panel = node("div","findings-panel");panel.append(node("h4",null,"复核要点"));
  findings.forEach((finding) => {
    const row = node("div","finding-row");row.append(badge(finding.status || "REVIEW",true));
    const text = node("p");if (finding.title) text.append(node("strong",null,finding.title));
    if (finding.message) text.append(document.createTextNode(typeof finding.message === "object" ? JSON.stringify(finding.message) : String(finding.message)));
    row.append(text);panel.append(row);
  });
  replace(dom.findingsPanel,[panel]);
}
function artifactURL(value) {
  if (typeof value !== "string" || !value) return null;
  try {
    const url = new URL(value,location.origin);
    if (url.origin !== location.origin || !url.pathname.startsWith("/runs/")) return null;
    return url.pathname + url.search;
  } catch { return null; }
}
function renderArtifacts(job) {
  const artifacts = (Array.isArray(job?.artifacts) ? job.artifacts : []).map((item) => ({...item,safeURL:artifactURL(item.url)})).filter((item) => item.safeURL);
  dom.artifactsSection.hidden = artifacts.length === 0;
  replace(dom.artifactsGrid,artifacts.map((item) => {
    const name = item.name || item.filename || "证据文件";
    const link = node("a","artifact-link");link.href = item.safeURL;
    if (/\.(html?|pdf)$/i.test(name)) { link.target = "_blank";link.rel = "noopener noreferrer"; }
    else link.download = String(name).split(/[\\/]/).pop();
    link.append(node("span","artifact-symbol",/\.html?$/i.test(name) ? "▤" : /\.(zip|tar|gz)$/i.test(name) ? "▣" : "▧"),node("span","artifact-name",name),node("span","artifact-arrow",/\.(html?|pdf)$/i.test(name) ? "↗" : "↓"));
    return link;
  }));
}
function renderJobDetail() {
  const job = jobs().find((item) => item.id === selectedJobId) || (activeJob()?.id === selectedJobId ? activeJob() : null);
  dom.stopButton.hidden = !job || rawStatus(job.status) !== "RUNNING" || String(job.status).toUpperCase() === "EXPORTING";
  dom.stopButton.disabled = pendingMutation || !connected || state?.demo || state?.read_only;
  dom.copyLogsButton.disabled = !job || !job.log_tail;
  if (!job) {
    dom.jobTitle.textContent = "尚无运行任务";dom.jobMeta.textContent = "每次测试使用独立目录保存证据";dom.jobOverview.replaceChildren();
    dom.logOutput.textContent = "等待测试启动。此处只显示实际执行输出。";
    renderStages(null);renderIdentity(null);renderTelemetry(null);renderFindings(null);renderArtifacts(null);return;
  }
  dom.jobTitle.textContent = profile(job.profile).title || job.profile;
  dom.jobMeta.textContent = `${job.id} · 开始 ${dateLabel(job.started_at,true)}${job.ended_at ? ` · 结束 ${dateLabel(job.ended_at,true)}` : ""}`;
  const status = jobStatus(job);
  const overview = node("div","job-overview");overview.append(node("span","job-status-label","本次技术检查"),badge(status));
  const message = status === "RUNNING" ? "正在执行实际负载，请保持主机与服务在线。" : status === "PASS" ? "本次已要求的技术检查通过；请复核实际测试范围。采购验收仍未完成。" : status === "CONSISTENT" ? "本次采集的身份信号一致；采购验收仍未完成，厂商与实物核验需另行完成。" : status === "FAIL" ? "存在失败或异常，请检查原始日志与报告。" : status === "STOPPED" ? "任务已停止，已采集的证据仍可复核。" : "运行证据需要复核，未完成项目不能作为通过。";
  overview.append(node("p",null,message));
  const stages = stagesFor(job);
  const finished = stages.filter((stage) => !["RUNNING","PENDING","UNKNOWN"].includes(rawStatus(stage.status))).length;
  if (stages.length) overview.append(node("span","job-progress",`${finished} / ${stages.length} 已记录阶段结束`));
  const acceptance = acceptanceFor(job);
  const purchase = node("div",`job-acceptance-note${acceptance.status === "BLOCKED" ? " blocked" : ""}`);
  purchase.append(node("strong",null,acceptance.status === "BLOCKED" ? "采购验收：存在阻断项" : "采购验收：尚未完成"),node("span",null,acceptance.statement));
  overview.append(purchase);
  replace(dom.jobOverview,[overview]);
  const log = Array.isArray(job.log_tail) ? job.log_tail.join("\n") : String(job.log_tail || "尚无输出。等待实际运行日志。");
  const atBottom = dom.logOutput.scrollHeight - dom.logOutput.scrollTop - dom.logOutput.clientHeight < 40;
  if (dom.logOutput.textContent !== log) {
    dom.logOutput.textContent = log;
    if (atBottom) dom.logOutput.scrollTop = dom.logOutput.scrollHeight;
  }
  renderStages(job);renderIdentity(job);renderTelemetry(job);renderFindings(job);renderArtifacts(job);
}

function render() {
  dom.demoBanner.hidden = !state?.demo;
  const parts = location.host;
  dom.endpoint.textContent = parts || "127.0.0.1:8765";
  dom.connectionStatus.classList.toggle("connected",connected);
  dom.connectionStatus.classList.toggle("disconnected",!connected);
  dom.connectionStatus.lastElementChild.textContent = connected ? "主机已连接" : "主机未连接";
  selectAutomaticJob();renderHardware();renderReadiness();renderProfiles();renderSelected();renderJobHistory();renderJobDetail();renderWorkflow();renderAcceptance();
}
async function refresh(force = false) {
  if (refreshing) return;
  refreshing = true;dom.refreshButton.disabled = true;
  try {
    const next = await api(`/api/state${force ? "?refresh=1" : ""}`);
    if (!next || typeof next !== "object" || !Array.isArray(next.jobs)) throw new Error("服务返回的状态格式不完整，请确认前后端版本一致。");
    state = next;connected = true;dom.connectionError.hidden = true;
    dom.lastUpdated.textContent = `最近刷新 ${new Date().toLocaleTimeString("zh-CN",{hour12:false})}`;
  } catch (error) {
    connected = false;
    dom.connectionError.textContent = `连接中断：${error.message} ${state ? "当前保留的是上次读取的数据，不能视为实时状态。" : "请先在 Ubuntu 主机启动服务，再建立 SSH 隧道。"}`;
    dom.connectionError.hidden = false;
    dom.lastUpdated.textContent = "连接中断 · 数据尚未刷新";
  } finally {
    refreshing = false;dom.refreshButton.disabled = false;render();
    clearTimeout(nextPoll);nextPoll = setTimeout(() => refresh(),document.hidden ? 7000 : connected ? 2000 : 5000);
  }
}
async function startJob(requestedProfile = selectedProfile) {
  if (!gate(requestedProfile).ok) return;
  pendingMutation = true;dom.actionMessage.hidden = true;renderSelected();renderReadiness();
  try {
    if (!csrfToken) await session();
    const payload = {profile:requestedProfile};
    if (requestedProfile === "burn") payload.seconds = Number(dom.burnSeconds.value);
    const result = await api("/api/jobs",{method:"POST",headers:{"Content-Type":"application/json","X-CSRF-Token":csrfToken},body:JSON.stringify(payload)});
    if (!result.id) throw new Error("主机未返回任务编号，请刷新后检查运行记录。");
    selectedJobId = result.id;selectionIsManual = false;
    showAction(`任务 ${result.id} 已启动。可在下方查看实际阶段与日志。`);
    dom.evidence.scrollIntoView({behavior:matchMedia("(prefers-reduced-motion: reduce)").matches ? "auto" : "smooth",block:"start"});
    await refresh();
  } catch (error) { showAction(`启动失败：${error.message}`,true); }
  finally { pendingMutation = false;renderSelected();renderJobDetail();renderReadiness(); }
}
async function stopJob() {
  const job = jobs().find((item) => item.id === selectedJobId);
  if (!job || rawStatus(job.status) !== "RUNNING" || pendingMutation || state?.demo || state?.read_only) return;
  pendingMutation = true;renderSelected();renderJobDetail();
  try {
    if (!csrfToken) await session();
    await api(`/api/jobs/${encodeURIComponent(job.id)}/stop`,{method:"POST",headers:{"X-CSRF-Token":csrfToken}});
    showAction("已请求停止任务。等待负载退出、容器清理与证据报告写入完成。");
    await refresh();
  } catch (error) { showAction(`停止请求失败：${error.message}`,true); }
  finally { pendingMutation = false;renderSelected();renderJobDetail(); }
}
async function copyText(text) {
  try {
    if (navigator.clipboard && window.isSecureContext) await navigator.clipboard.writeText(text);
    else {
      const field = node("textarea");field.value = text;field.style.position = "fixed";field.style.left = "-10000px";
      document.body.append(field);field.select();const success = document.execCommand("copy");field.remove();
      if (!success) throw new Error("copy unavailable");
    }
    toast("已复制");
  } catch { toast("无法自动复制，请选中文本后手动复制。"); }
}

dom.startButton.addEventListener("click",() => startJob(selectedProfile));
dom.importImageButton.addEventListener("click",() => startJob("import-image"));
dom.stopButton.addEventListener("click",stopJob);
dom.refreshButton.addEventListener("click",() => refresh(true));
dom.copyLogsButton.addEventListener("click",() => copyText(dom.logOutput.textContent));
document.querySelectorAll(".copy-code").forEach((button) => button.addEventListener("click",() => copyText(button.parentElement.querySelector("code").textContent)));
document.querySelectorAll(".nav-link").forEach((link) => link.addEventListener("click",() => { document.querySelectorAll(".nav-link").forEach((other) => other.classList.remove("active"));link.classList.add("active"); }));
if ("IntersectionObserver" in window) {
  const observer = new IntersectionObserver((entries) => {
    const visible = entries.filter((entry) => entry.isIntersecting).sort((a,b) => a.boundingClientRect.top - b.boundingClientRect.top)[0];
    if (!visible) return;
    document.querySelectorAll(".nav-link").forEach((link) => link.classList.toggle("active",link.hash === `#${visible.target.id}`));
  },{rootMargin:"-15% 0px -60% 0px",threshold:0});
  ["overview","workflow","evidence","guide"].forEach((id) => observer.observe(dom[id]));
}
document.addEventListener("visibilitychange",() => { if (!document.hidden) refresh(); });
render();
session().catch((error) => { showAction(`操作会话尚未就绪：${error.message}`,true); }).finally(() => refresh());
