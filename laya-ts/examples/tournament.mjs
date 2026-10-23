// Run after scripts/tournament_parity.py; all model files must already be local.
// node laya-ts/examples/tournament.mjs --model-dir SPLIT_EXPORT \
//   --fixture /tmp/tournament-python.json --report /tmp/tournament-node.json
// Java contributes the input scenarios only. This runs the TS split-ONNX model.
import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { readFile, stat, mkdir, writeFile } from "node:fs/promises";
import { dirname, isAbsolute, relative, resolve, sep } from "node:path";
import { parseArgs } from "node:util";
import { Agent, predictTournament } from "../dist/index.js";

const { values } = parseArgs({ options: {
  "model-dir": { type: "string" }, fixture: { type: "string" },
  report: { type: "string" }, tolerance: { type: "string", default: "0.001" },
} });
assert(values["model-dir"] && values.fixture && values.report,
  "usage: tournament.mjs --model-dir LOCAL_SPLIT_EXPORT --fixture PYTHON.json --report NODE.json");
const modelDir = resolve(values["model-dir"]);
assert((await stat(modelDir)).isDirectory(), "--model-dir must be an existing local directory");
const reportPath = resolve(values.report);
assert.notEqual(resolve(values.fixture), reportPath, "report must not overwrite fixture");
const reportRelative = relative(modelDir, reportPath);
assert(isAbsolute(reportRelative) || reportRelative.startsWith(`..${sep}`),
  "--report must be outside the model directory");
const tolerance = Number(values.tolerance);
assert(Number.isFinite(tolerance) && tolerance > 0, "tolerance must be a positive finite number");
const fixtureBytes = await readFile(values.fixture);
const fixture = JSON.parse(fixtureBytes.toString("utf8"));
assert.equal(fixture.schema, "laya-ts-tournament-parity/1");
assert(Number.isInteger(fixture.repeats) && fixture.repeats > 0, "invalid repeat count");
assert(Array.isArray(fixture.cases) && fixture.cases.length > 0, "fixture has no cases");
assert.equal(new Set(fixture.cases.map(item => item.name)).size, fixture.cases.length,
  "fixture case names must be unique");
for (const item of fixture.cases) {
  assert(typeof item.name === "string" && item.name.length > 0, "invalid case name");
  assert.equal(item.plain?.calls?.length, 1, "plain reference must contain one real call");
  assert(Array.isArray(item.tournament?.calls) && item.tournament.calls.length > 0,
    "tournament reference has no calls");
}
for (const name of ["encoder.onnx", "head.onnx", "rl_agent_config.json", "tokenizer.json"]) {
  assert.match(fixture.onnx_sha256?.[name] ?? "", /^[0-9a-f]{64}$/, `missing digest: ${name}`);
}
const pythonOptions = fixture.predict_kwargs;
assert.deepEqual(Object.keys(pythonOptions).sort(), ["head_max_len", "lang", "max_len", "min_confidence"]);
const options = { lang: pythonOptions.lang, maxLen: pythonOptions.max_len,
  headMaxLen: pythonOptions.head_max_len, minConfidence: pythonOptions.min_confidence };

function recording(agent) {
  return { calls: [], async predict(state, questions, kwargs = {}) {
    const call = { state: structuredClone(state), questions: structuredClone(questions),
      kwargs: { lang: kwargs.lang, max_len: kwargs.maxLen, head_max_len: kwargs.headMaxLen,
        min_confidence: kwargs.minConfidence } };
    const started = performance.now();
    const result = await agent.predict(state, questions, kwargs);
    this.calls.push({ ...call, answers: structuredClone(result.answers),
      usage: structuredClone(result.usage), elapsed_ms: performance.now() - started });
    return result;
  } };
}

async function measure(agent, run) {
  await run(recording(agent)); // warm this path, outside the measured samples
  const elapsed = [];
  let calls, result;
  for (let i = 0; i < fixture.repeats; i++) {
    const recorder = recording(agent);
    const started = performance.now();
    result = await run(recorder);
    elapsed.push(performance.now() - started);
    if (calls === undefined) calls = recorder.calls;
    else assert.deepEqual(recorder.calls.map(c => c.answers), calls.map(c => c.answers),
      "TS outputs changed across repeated calls");
  }
  const sorted = [...elapsed].sort((a, b) => a - b);
  const mid = Math.floor(sorted.length / 2);
  const median = sorted.length % 2 ? sorted[mid] : (sorted[mid - 1] + sorted[mid]) / 2;
  return { calls, result, model_calls: calls.length, elapsed_ms: elapsed, median_ms: median };
}

const errors = [];
let maxDifference = 0;
function compare(expected, actual, path) {
  if (typeof expected === "number") {
    const difference = Math.abs(expected - actual);
    if (typeof actual !== "number" || !Number.isFinite(actual) || difference > tolerance) {
      errors.push(`${path}: expected ${expected}, got ${actual}`);
    }
    if (Number.isFinite(difference)) maxDifference = Math.max(maxDifference, difference);
  } else if (expected && typeof expected === "object") {
    if (!actual || typeof actual !== "object" ||
        JSON.stringify(Object.keys(expected)) !== JSON.stringify(Object.keys(actual))) {
      errors.push(`${path}: object keys or their order differ`);
      return;
    }
    for (const key of Object.keys(expected)) compare(expected[key], actual[key], `${path}.${key}`);
  } else if (expected !== actual) errors.push(`${path}: expected ${expected}, got ${actual}`);
}

function compareAnswers(expected, actual, path) {
  if (JSON.stringify(Object.keys(expected)) !== JSON.stringify(Object.keys(actual ?? {}))) {
    errors.push(`${path}: answer IDs or their order differ`);
    return;
  }
  for (const [qid, answer] of Object.entries(expected)) {
    // Runtime-specific model names and abstention annotations are outside this port.
    for (const field of ["type", "choice", "noul", "score", "probabilities",
      "confidence", "answer_confidence", "action"]) {
      if (field in answer) compare(answer[field], actual[qid]?.[field], `${path}.${qid}.${field}`);
    }
    compare(answer.low_confidence === true, actual[qid]?.low_confidence === true,
      `${path}.${qid}.low_confidence`);
  }
}

function compareRun(expected, actual, path) {
  if (expected.calls.length !== actual.calls.length) errors.push(`${path}: model call count differs`);
  for (let i = 0; i < Math.min(expected.calls.length, actual.calls.length); i++) {
    const want = expected.calls[i], got = actual.calls[i];
    // Order is significant for state serialization, options, and generated group IDs.
    for (const field of ["state", "questions", "kwargs"]) {
      if (JSON.stringify(want[field]) !== JSON.stringify(got[field])) {
        errors.push(`${path}.calls[${i}].${field}: exact input mismatch`);
      }
    }
    compareAnswers(want.answers, got.answers, `${path}.calls[${i}].answers`);
  }
  compareAnswers(expected.result.answers, actual.result.answers, `${path}.final.answers`);
  if (expected.result.tournament) {
    // Counts and finalist order are exact, not floating-point comparisons.
    if (JSON.stringify(expected.result.tournament) !== JSON.stringify(actual.result.tournament)) {
      errors.push(`${path}.final.tournament: metadata differs`);
    }
  }
}

const agent = await Agent.load(modelDir, { device: "cpu", numThreads: 2,
  expectedSha256: fixture.onnx_sha256 });
const results = [];
for (const item of fixture.cases) {
  const { state, questions } = item;
  const plain = await measure(agent, recorder => recorder.predict(state, questions, options));
  const tournament = await measure(agent, recorder =>
    predictTournament(recorder, state, questions, item.group_size ?? 16, options));
  compareRun(item.plain, plain, `${item.name}.plain`);
  compareRun(item.tournament, tournament, `${item.name}.tournament`);
  results.push({ name: item.name, plain, tournament });
  console.log(JSON.stringify({ case: item.name, plain_ms: plain.median_ms,
    tournament_ms: tournament.median_ms, model_calls: tournament.model_calls }));
}
const report = { schema: "laya-ts-tournament-parity-result/1",
  fixture_sha256: createHash("sha256").update(fixtureBytes).digest("hex"),
  onnx_sha256: fixture.onnx_sha256, environment: { node: process.version,
    platform: process.platform, arch: process.arch, device: "cpu", num_threads_requested: 2 },
  tolerance, max_absolute_difference: maxDifference, passed: errors.length === 0,
  errors, cases: results,
  timing_note: "One warmup per case and strategy; elapsed samples exclude loading and hashing. Small parity experiment, not an accuracy or throughput benchmark." };
await mkdir(dirname(reportPath), { recursive: true });
await writeFile(reportPath, JSON.stringify(report, null, 2) + "\n", "utf8");
console.log(JSON.stringify({ passed: report.passed, max_absolute_difference: maxDifference,
  errors: errors.length, report: reportPath }));
if (errors.length) process.exitCode = 1;
