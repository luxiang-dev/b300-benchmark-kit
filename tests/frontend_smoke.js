"use strict";

// Synthetic DOM/API fixtures only. This test never executes GPU or Docker work.
// Run: node tests/frontend_smoke.js (Node 18+; no npm dependencies).
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const vm = require("node:vm");
const root = path.resolve(__dirname, "..");
const html = fs.readFileSync(path.join(root, "dashboard/index.html"), "utf8");
const js = fs.readFileSync(path.join(root, "dashboard/app.js"), "utf8");
const ids = [...html.matchAll(/\bid="([^"]+)"/g)].map((match) => match[1]);
assert.equal(new Set(ids).size, ids.length, "HTML IDs must be unique");
for (const match of js.matchAll(/\bdom\.([A-Za-z_][\w]*)/g)) {
  assert.ok(ids.includes(match[1]), `Missing HTML element: ${match[1]}`);
}

class Element {
  constructor(tag = "div") {
    this.tagName = tag;
    this.children = [];
    this.style = {};
    this.classList = {toggle() {}, add() {}, remove() {}};
    this.value = "3600";
    this.scrollHeight = this.scrollTop = this.clientHeight = 0;
    this._text = "";
    this.attributes = {};
  }
  set textContent(value) { this._text = String(value); this.children = []; }
  get textContent() { return this._text; }
  get lastElementChild() { return this.children.at(-1); }
  append(...children) { this.children.push(...children); }
  replaceChildren(...children) { this.children = children; }
  setAttribute(key, value) { this.attributes[key] = value; }
  addEventListener() {}
  scrollIntoView() {}
}
const elements = ids.map((id) => Object.assign(new Element(), {id}));
const byId = Object.fromEntries(elements.map((element) => [element.id, element]));
byId.connectionStatus.append(new Element("span"), new Element("span"));
const document = {
  getElementById: (id) => byId[id],
  querySelectorAll: (selector) => selector === "[id]" ? elements : [],
  createElement: (tag) => new Element(tag),
  createElementNS: (_, tag) => new Element(tag),
  createTextNode: (text) => ({textContent: String(text)}),
  addEventListener() {},
  hidden: false,
};
const mock = {enabled: false, requests: [], state: null};
const context = vm.createContext({
  document, location: {origin: "http://127.0.0.1:8765", host: "127.0.0.1:8765"},
  window: {}, navigator: {}, AbortController, URL,
  setTimeout: () => 0, clearTimeout() {}, matchMedia: () => ({matches: true}),
  fetch: async (url, options = {}) => {
    if (!mock.enabled) return new Promise(() => {});
    mock.requests.push({url, options});
    let body;
    if (url === "/api/session") body = {csrf_token: "synthetic-test-token"};
    else if (url === "/api/jobs" && options.method === "POST") {
      const job = {id: "fixture-run", profile: JSON.parse(options.body).profile, status: "RUNNING", stages: [], log_tail: "test-only output", artifacts: []};
      mock.state = {...mock.state, busy: true, active_job: job, jobs: [job]};
      body = {id: job.id};
    } else if (url === "/api/jobs/fixture-run/stop") {
      mock.state.jobs[0].status = "CANCELLED";
      mock.state.busy = false; mock.state.active_job = null; body = {stopped: true};
    } else if (url.startsWith("/api/state")) body = mock.state;
    else throw new Error(`Unexpected synthetic request: ${url}`);
    return {ok: true, status: 200, json: async () => body};
  },
});
vm.runInContext(js, context, {filename: "dashboard/app.js"});
const evaluate = (code) => vm.runInContext(code, context);
const base = {
  platform: "linux", busy: false, jobs: [], active_job: null, profiles: [],
  readiness: {
    gpus: Array.from({length: 8}, (_, index) => ({index, name: "NVIDIA B300", uuid: `GPU-${index}`, memory_mib: 270000, ecc: "Enabled", mig: "Disabled"})),
    tools: {python3: true, docker: true, nvidia_smi: true}, image: {available: true},
    bundle: {sealed: true}, deepseek: {configured: true}, warnings: [],
  },
};
function setState(value) {
  mock.state = structuredClone(value);
  evaluate(`state=${JSON.stringify(value)}; connected=true; pendingMutation=false; selectionIsManual=false; render();`);
}

async function main() {
  setState(base);
  assert.equal(evaluate("gate('authenticity').ok"), true);
  assert.equal(evaluate("gate('deepseek').ok"), true, "Readiness uses the backend configured field");
  evaluate("state.busy=true");
  assert.equal(evaluate("gate('authenticity').ok"), false, "No concurrent job");
  assert.equal(evaluate("gate('identity', true).ok"), true, "Readiness stays recognized during a job");
  for (const flag of ["demo", "read_only"]) {
    setState({...base, [flag]: true});
    assert.equal(evaluate("gate('identity').ok"), false, `${flag} is read-only`);
    assert.equal(byId.startButton.disabled, true);
  }
  setState({...base, readiness: {...base.readiness, gpus: [], image: {available: false}, tools: {...base.readiness.tools, docker: false}}});
  assert.equal(evaluate("gate('identity').ok"), true, "Identity can collect missing/contradictory evidence without a GPU image");
  assert.equal(evaluate("gate('authenticity').ok"), false, "Heavy workloads require eight cards");
  setState({...base, readiness: {...base.readiness, gpus: [{name: "other GPU"}]}});
  assert.equal(evaluate("gate('identity').ok"), true, "Unexpected identity must remain collectable");
  assert.equal(evaluate("gate('authenticity').ok"), false);
  assert.equal(evaluate("gate('import-image').ok"), true, "Image import needs no GPU identity");
  setState({...base, profiles: [{id: "import-image", available: false, unavailable_reason: "missing archive"}]});
  assert.equal(evaluate("gate('import-image').ok"), false, "Honor backend archive eligibility");
  setState(base);
  for (const url of ["https://example.org/file", "javascript:alert(1)", "/docs/file", "//example.org/runs/file"]) {
    assert.equal(evaluate(`artifactURL(${JSON.stringify(url)})`), null);
  }
  assert.equal(evaluate("artifactURL('/runs/id/report.html')"), "/runs/id/report.html");
  assert.equal(evaluate("jobStatus({status:'completed', exit_code:0})"), "UNKNOWN", "Exit zero alone is no hardware PASS");
  assert.equal(evaluate("jobStatus({status:'completed', summary:{overall:'REVIEW'}})"), "REVIEW");
  assert.equal(evaluate("jobStatus({status:'EXPORTING', summary:{overall:'PASS'}})"), "RUNNING", "Evidence export is still running");
  assert.equal(evaluate("stagesFor({stages:[{name:'memory',status:'RUNNING'},{name:'memory',status:'PASS'}]}).length"), 1);
  assert.equal(evaluate("findingsFrom({identity_summary:{findings:[{severity:'FAIL',code:'invalid',detail:'failure'}]}})[0].status"), "FAIL");

  const evidenceJob = {id: "fixture", profile: "identity", status: "REVIEW", log_tail: "<script>synthetic()</script>", artifacts: [], stages: [],
    identity_summary: {status: "CONSISTENT", vendor_confirmation: "NOT_PERFORMED", per_gpu: [{host_index: 0, name: "NVIDIA B300", capability: [10, 3], status: "CONSISTENT"}]},
    telemetry: {per_gpu: {}}, summary: {overall: "REVIEW", telemetry: {per_gpu: {"GPU-0": {samples: 30, peak_temperature_c: 50}}}}};
  setState({...base, jobs: [evidenceJob]});
  assert.equal(byId.logOutput.textContent, evidenceJob.log_tail, "Log HTML is literal text");
  assert.equal(byId.identityPanel.hidden, false);
  assert.equal(byId.telemetryPanel.hidden, false, "Historical report telemetry survives empty live data");

  setState(base); mock.enabled = true; mock.requests = [];
  await evaluate("csrfToken=null; selectedProfile='identity'; startJob()");
  const start = mock.requests.find((request) => request.url === "/api/jobs");
  assert.ok(start, "Start posts a concrete job");
  assert.equal(start.options.headers["X-CSRF-Token"], "synthetic-test-token");
  assert.equal(start.options.credentials, "same-origin");
  assert.equal(JSON.parse(start.options.body).profile, "identity");
  await evaluate("stopJob()");
  const stop = mock.requests.find((request) => request.url.endsWith("/stop"));
  assert.ok(stop);
  assert.equal(stop.options.headers["X-CSRF-Token"], "synthetic-test-token");
  const mutations = mock.requests.filter((request) => request.options.method === "POST").length;
  evaluate("state.demo=true");
  await evaluate("startJob()");
  assert.equal(mock.requests.filter((request) => request.options.method === "POST").length, mutations, "Read-only actions never POST");
  console.log("Frontend smoke tests passed: DOM IDs, real-data states, eligibility, safe logs/links, CSRF, start/stop, and read-only guards.");
}
main().catch((error) => { console.error(error); process.exitCode = 1; });
