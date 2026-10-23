import { readFileSync } from "node:fs";
import { describe, expect, it, vi } from "vitest";
import { Agent, Router, DEFAULT_TOURNAMENT_GROUP, predictTournament } from "../src/index.js";
import type { QuestionDef, SessionProvider } from "../src/index.js";

type Questions = Record<string, QuestionDef>;
const labels = (q: QuestionDef): string[] => Array.isArray(q.criteria)
  ? q.criteria.map(String) : Object.keys(q.criteria as object);
const choice = (criteria: unknown): QuestionDef => ({ type: "choice", instructions: "Choose a route.", criteria });
const names = (n: number): string[] => Array.from({ length: n }, (_, i) => `label-${i}`);

class Ranker {
  calls: { state: unknown; questions: Questions; opts: Record<string, unknown> | undefined }[] = [];
  returned: Record<string, any>[] = [];

  constructor(readonly better: (a: string, b: string) => boolean = () => false) {}

  async predict(state: unknown, questions: Questions, opts?: Record<string, unknown>) {
    this.calls.push({ state, questions, opts });
    const answers = Object.fromEntries(Object.entries(questions).map(([qid, q]) => {
      if (q.type !== "choice") return [qid, { type: q.type, score: 0.5, noul: 0.5 }];
      const options = labels(q);
      const winner = options.reduce((best, next) => this.better(next, best) ? next : best);
      return [qid, { type: "choice", choice: winner, probabilities: { [winner]: 1 }, confidence: 1 }];
    }));
    const result = {
      model: "fixture", answers, usage: { input_tokens: this.calls.length, output_tokens: 0 },
      routing: { model: "fixture", reason: "fixture route" }, trace: this.calls.length,
    };
    this.returned.push(result);
    return result;
  }
}

// Reuse the existing Python-generated oracle, not another copy of the grouping algorithm.
// No Java toolchain or checkpoint is needed to read these recorded calls.
const golden = JSON.parse(readFileSync(new URL("../../laya-java/fixtures/tournament.json", import.meta.url), "utf8")) as {
  default_group_size: number;
  rank_probe: Record<string, string>;
  splits: { group_size: number; n: number; rounds: number[][]; finalists: string[]; choice: string }[];
};
function rank(label: string): bigint {
  let value = 0xcbf29ce484222325n;
  for (const byte of new TextEncoder().encode(`rank:${label}`)) {
    value = BigInt.asUintN(64, (value ^ BigInt(byte)) * 0x100000001b3n);
  }
  return value;
}

describe("tournament decisions", () => {
  it("uses the reference default and the independently recorded rank function", () => {
    expect(DEFAULT_TOURNAMENT_GROUP).toBe(16);
    expect(DEFAULT_TOURNAMENT_GROUP).toBe(golden.default_group_size);
    expect(golden.splits.length).toBeGreaterThanOrEqual(100);
    for (const [label, value] of Object.entries(golden.rank_probe)) expect(String(rank(label))).toBe(value);
  });

  it.each(golden.splits)("matches Python's recorded rounds for $n labels / groups of $group_size", async (row) => {
    const options = Array.from({ length: row.n }, (_, i) => `l${String(i).padStart(4, "0")}`);
    const runner = new Ranker((a, b) => rank(a) < rank(b));
    const result = await predictTournament(runner, "state", { q: choice(options) }, row.group_size);
    expect(runner.calls.slice(0, -1).map((c) => Object.values(c.questions).map((q) => labels(q).length)))
      .toEqual(row.rounds);
    expect(result.tournament.q).toEqual({ labels: row.finalists, n: row.n, rounds: row.rounds.length });
    expect(result.answers.q).toMatchObject({ choice: row.choice });
  });

  it.each([77, 257])("a consistently best label survives from every position among %i options", async (n) => {
    const options = names(n);
    for (const best of [options[0], options[Math.floor(n / 2)], options[n - 1]]) {
      const runner = new Ranker((a) => a === best);
      const result = await predictTournament(runner, "state", { q: choice(options) });
      expect(result.answers.q).toMatchObject({ choice: best });
      expect(runner.calls).toHaveLength(n === 77 ? 2 : 3);
      if (n === 77) {
        expect(Object.values(runner.calls[0].questions).map((q) => labels(q).length)).toEqual([15, 15, 16, 15, 16]);
      }
      const firstRound = Object.values(runner.calls[0].questions).flatMap(labels);
      expect(firstRound).toEqual(options);
      expect(new Set(firstRound).size).toBe(n);
    }
  });

  it("combines differently sized questions into rounds and preserves the final result", async () => {
    const big = choice(Object.fromEntries(names(20).map((label) => [label, { description: label }])));
    const medium = choice(["m0", "m1", "m2", "m3", "m4"]);
    const small = choice(["x", "y"]);
    const score: QuestionDef = { type: "score", instructions: "Rate urgency.", criteria: ["low", "high"] };
    const questions = { big, medium, small, score };
    const original = structuredClone(questions);
    const state = { text: "request", turns: ["original"] };
    const onPredictStart = vi.fn();
    const opts = { model: "english", lang: "de", hooks: [{ onPredictStart }], minConfidence: 0.8, headMaxLen: 160 };
    const runner = new Ranker();
    const result = await predictTournament(runner, state, questions, 4, opts);
    expect(runner.calls).toHaveLength(3);
    expect(runner.calls.map((c) => Object.keys(c.questions).length)).toEqual([7, 2, 4]);
    for (const call of runner.calls) {
      expect(call.state).toBe(state);
      expect(call.opts).toEqual(opts);
      expect(call.opts?.hooks).toBe(opts.hooks);
    }
    expect(result.tournament.big).toEqual({ labels: ["label-0", "label-8"], n: 20, rounds: 2 });
    expect(result.tournament.medium).toEqual({ labels: ["m0", "m2"], n: 5, rounds: 1 });
    expect(result.tournament.small).toEqual({ labels: ["x", "y"], n: 2, rounds: 0 });
    const final = runner.calls.at(-1)!.questions;
    expect(final.small).toBe(small);
    expect(final.score).toBe(score);
    expect(final.big.criteria).toEqual({ "label-0": { description: "label-0" }, "label-8": { description: "label-8" } });
    expect(questions).toEqual(original);
    const returned = runner.returned.at(-1)!;
    expect(result).toEqual({ ...returned, tournament: result.tournament });
    expect(result.answers).toBe(returned.answers);
    expect(result.usage).toBe(returned.usage);
    expect(returned).not.toHaveProperty("tournament");
  });

  it.each([{}, { small: choice(["x", "y"]), flag: { type: "noul", instructions: "Is it urgent?" } }])(
    "uses just one final call for an empty or already small request", async (questions) => {
      const runner = new Ranker();
      await predictTournament(runner, "state", questions as Questions);
      expect(runner.calls).toHaveLength(1);
      expect(runner.calls[0].questions).toEqual(questions);
    },
  );

  it("awaits a round before asking finalists, and preserves rejection identity", async () => {
    let resolve!: (value: unknown) => void;
    const blocked = new Promise((done) => { resolve = done; });
    const predict = vi.fn().mockReturnValueOnce(blocked).mockResolvedValue({ answers: {}, usage: {} });
    const pending = predictTournament({ predict }, "state", { q: choice(names(5)) }, 4);
    await Promise.resolve();
    expect(predict).toHaveBeenCalledTimes(1);
    resolve({ answers: { "0": { type: "choice", choice: "label-0" }, "1": { type: "choice", choice: "label-2" } } });
    await pending;
    expect(predict).toHaveBeenCalledTimes(2);
    const error = new Error("provider stopped");
    const reject = vi.fn().mockRejectedValue(error);
    await expect(predictTournament({ predict: reject }, "state", { q: choice(names(20)) }, 2)).rejects.toBe(error);
    expect(reject).toHaveBeenCalledTimes(1);
  });

  it.each(["systemOne", "system_one"])("supports the %s fallback with its receiver", async (method) => {
    const runner = new Ranker();
    const adapter = { owner: runner, [method]: function (this: { owner: Ranker }, ...args: Parameters<Ranker["predict"]>) {
      return this.owner.predict(...args);
    } };
    await predictTournament(adapter, "state", { q: choice(names(5)) }, 4);
    expect(runner.calls).toHaveLength(2);
  });

  it("keeps special question IDs and labels as own entries", async () => {
    const options = ["__proto__", "constructor", "toString", "safe", "last"];
    const questions = Object.fromEntries(["__proto__", "constructor", "toString"].map((qid) => [qid, choice(options)]));
    const runner = new Ranker();
    const result = await predictTournament(runner, "state", questions, 4);
    expect(Object.keys(result.tournament)).toEqual(Object.keys(questions));
    for (const qid of Object.keys(questions)) {
      expect(Object.hasOwn(result.tournament, qid)).toBe(true);
      expect(result.tournament[qid].labels).toEqual(["__proto__", "toString"]);
      expect(Object.hasOwn(runner.calls.at(-1)!.questions, qid)).toBe(true);
    }
    const dictRunner = new Ranker();
    await predictTournament(dictRunner, "state", { q: choice(Object.fromEntries(options.map((label) => [label, label]))) }, 4);
    expect(Object.keys(dictRunner.calls.at(-1)!.questions.q.criteria as object)).toEqual(["__proto__", "toString"]);
    expect(({} as Record<string, unknown>).polluted).toBeUndefined();
  });

  it("uses native object enumeration for numeric keys and preserves array labels", async () => {
    const criteria = Object.fromEntries(["10", "2", "1", "z", "a"].map((k) => [k, `about ${k}`]));
    const runner = new Ranker();
    await predictTournament(runner, "state", { q: choice(criteria) }, 4);
    expect(Object.values(runner.calls[0].questions).flatMap(labels)).toEqual(["1", "2", "10", "z", "a"]);
    const arrays = new Ranker();
    const result = await predictTournament(arrays, "state", { q: choice([3, false, "x", 9, "z"]) }, 4);
    expect(result.tournament.q.labels).toEqual(["3", "x"]);
  });

  it("drops narrowed option_order once per question, preserving passthrough and the caller", async () => {
    const warn = vi.spyOn(console, "warn").mockImplementation(() => {});
    try {
      const big = { ...choice(names(20)), option_order: names(20).map((_, i) => 19 - i) };
      const small = { ...choice(["x", "y"]), option_order: [1, 0] };
      const before = structuredClone({ big, small });
      const runner = new Ranker();
      await predictTournament(runner, "state", { big, small }, 4);
      expect(warn).toHaveBeenCalledTimes(1);
      expect(warn.mock.calls[0].join(" ")).toContain("big");
      expect(warn.mock.calls[0].join(" ")).toContain("option_order");
      for (const call of runner.calls.slice(0, -1)) {
        for (const q of Object.values(call.questions)) expect(q).not.toHaveProperty("option_order");
      }
      expect(runner.calls.at(-1)!.questions.big).not.toHaveProperty("option_order");
      expect(runner.calls.at(-1)!.questions.small).toBe(small);
      expect({ big, small }).toEqual(before);
    } finally {
      warn.mockRestore();
    }
  });

  it.each([1, 0, -1, 2.5, NaN, Infinity, Number.MAX_SAFE_INTEGER + 1, true, "16", null])("rejects invalid group size %s before inference", async (groupSize) => {
    const predict = vi.fn();
    await expect(predictTournament({ predict }, "state", {}, groupSize as number)).rejects.toThrow();
    expect(predict).not.toHaveBeenCalled();
  });

  it.each([undefined, null, [], {}, "a", ["a", "a"], [1, "1"]].map((criteria) => ({ criteria })))("rejects invalid criteria $criteria before any group", async ({ criteria }) => {
    const predict = vi.fn();
    await expect(predictTournament({ predict }, "state", { valid: choice(names(20)), bad: choice(criteria) }, 4)).rejects.toThrow();
    expect(predict).not.toHaveBeenCalled();
  });

  it.each([null, [], "questions", 10, new Map(), new Set()].map((questions) => ({ questions })))("rejects values that are not question dictionaries: $questions", async ({ questions }) => {
    const predict = vi.fn();
    await expect(predictTournament({ predict }, "state", questions as Questions)).rejects.toThrow();
    expect(predict).not.toHaveBeenCalled();
  });

  it.each([[null, "b", "c"], [undefined, "b", "c"], [["nested"], "b", "c"], [{ route: "bad" }, "b", "c"]].map((criteria) => ({ criteria })))(
    "does not turn an invalid label into a valid string while narrowing $criteria", async ({ criteria }) => {
      const predict = vi.fn();
      await expect(predictTournament({ predict }, "state", { q: choice(criteria) }, 2)).rejects.toThrow();
      expect(predict).not.toHaveBeenCalled();
    },
  );

  it("accepts null-prototype dictionaries without losing their entries", async () => {
    const questions = Object.assign(Object.create(null), { q: choice(names(5)) });
    const runner = new Ranker();
    const result = await predictTournament(runner, "state", questions, 4);
    expect(result.tournament.q).toEqual({ labels: ["label-0", "label-2"], n: 5, rounds: 1 });
  });

  it.each([null, [], "not a result"].map((result) => ({ result })))("rejects final results that are not dictionaries: $result", async ({ result }) => {
    const predict = vi.fn().mockResolvedValue(result);
    await expect(predictTournament({ predict }, "state", { q: choice(["a"]) })).rejects.toThrow();
    expect(predict).toHaveBeenCalledTimes(1);
  });

  it.each([
    null, [], {}, { answers: {} }, { answers: { "0": { type: "choice", choice: "label-0" } } },
    { answers: { "0": { type: "choice", choice: "outside" }, "1": { type: "choice", choice: "label-2" } } },
    { answers: { "0": { type: "choice", choice: "label-2" }, "1": { type: "choice", choice: "label-2" } } },
    { answers: Object.create({ "0": { choice: "label-0" }, "1": { choice: "label-2" } }) },
    { answers: { "0": Object.create({ choice: "label-0" }), "1": { choice: "label-2" } } },
  ].map((output) => ({ output })))("stops instead of advancing malformed or foreign group winners $output", async ({ output }) => {
    const predict = vi.fn().mockResolvedValue(output);
    await expect(predictTournament({ predict }, "state", { q: choice(names(5)) }, 4)).rejects.toThrow();
    expect(predict).toHaveBeenCalledTimes(1);
  });
});

function realAgent() {
  const batches: number[] = [];
  const provider: SessionProvider = {
    async runEncoder(batch) {
      batches.push(batch.qtype.length);
      return { lastHidden: batch.inputIds.map((row) => row.map(() => [0])) };
    },
    async runHead(_hidden, batch) {
      return {
        logits: batch.markerMask.map((mask) => mask.map((_, i) => i === 0 ? 2 : 0)),
        act: batch.qtype.map(() => [3, 0]),
      };
    },
  };
  return { agent: new Agent({ provider }), batches };
}

describe("tournament runtime integration", () => {
  it("executes each round as one real Agent provider batch and exposes final-only probabilities", async () => {
    const { agent, batches } = realAgent();
    const starts = vi.fn();
    const result = await predictTournament(agent, "request", { q: choice(names(20)) }, 4, {
      onPredictStart: starts, minConfidence: 1,
    });
    expect(batches).toEqual([5, 2, 1]);
    expect(starts).toHaveBeenCalledTimes(3);
    expect(result.answers.q.type).toBe("choice");
    if (result.answers.q.type !== "choice") throw new Error("expected choice");
    expect(Object.keys(result.answers.q.probabilities)).toEqual(result.tournament.q.labels);
    expect(result.answers.q.low_confidence).toBe(true);
    expect(result.usage.input_tokens).toBeGreaterThan(0);
    expect(result.model).toBe("laya-rl-agent");
  });

  it("preserves Router routing and runs explicit model options and hooks in every round", async () => {
    const { agent, batches } = realAgent();
    const router = new Router();
    router.attach("english", agent);
    const routes: unknown[] = [];
    const hooks = { onPredictStart(ctx: { decision?: unknown }) { routes.push(ctx.decision); } };
    const result = await predictTournament(router, "退款", { q: choice(names(5)) }, 4, {
      model: "english", hooks, minConfidence: 1,
    });
    expect(batches).toEqual([2, 1]);
    expect(routes).toHaveLength(2);
    for (const route of routes) expect(route).toMatchObject({ model: "english" });
    expect(result).toHaveProperty("routing.model", "english");
    expect(result.answers.q.low_confidence).toBe(true);
  });
});
