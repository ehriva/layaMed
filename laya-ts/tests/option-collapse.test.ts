import { describe, expect, it } from "vitest";
import { Agent, defaultTokenizer } from "../src/agent.js";
import { buildSequence, collapsedOptions } from "../src/common.js";
import type { TokenizerLike } from "../src/tokenizer.js";

// TS mirror of tests/test_option_collapse.py (issue #538): options capped to fit the head budget
// can collapse into one token span, and the answer then comes from a question that can no longer
// name them apart. Python reports it as `build_sequence(..., return_stats=True)`'s third value and
// publishes it as `usage["options"]`; the TS port built the identical cap (same 48-token floor,
// same `max(4, (headMaxLen - 16) // n)` re-cap) but kept no count of what it cost.
//
// Every expected number below is what laya/common.py returns for the same stub tokenizer on
// `d8a2e59`, so a divergence here is a port bug rather than a style choice.

/** Character-level ids, so options that share a prefix share their leading tokens. */
const TOK: TokenizerLike = {
  ...defaultTokenizer(),
  clsId: 2,
  sepId: 3,
  maskId: 1,
  padId: 0,
  maskToken: "[MASK]",
  encode: (text: string) => [...text].map((c) => 10 + (c.charCodeAt(0) % 90)),
};

const qChoice = (labels: string[]) => ({
  t: "choice",
  ins: "which one?",
  crit: Object.fromEntries(labels.map((k) => [k, k.replace(/_/g, " ")])),
}) as never;

// Four labels that agree for their first fourteen characters, as MASSIVE's iot_hue_light* do.
const HUE = ["iot_hue_lightup", "iot_hue_lightoff", "iot_hue_lightdim", "iot_hue_lightchange"];
const MANY = Array.from({ length: 40 }, (_, i) => `label_${String(i).padStart(2, "0")}`);

const built = (labels: string[], maxLen: number, headMaxLen: number, state = "a state") =>
  buildSequence(TOK, state, qChoice(labels), maxLen, headMaxLen);
const statsFor = (labels: string[], maxLen: number, headMaxLen: number, state = "a state") =>
  built(labels, maxLen, headMaxLen, state).stats;

describe("the option report's shape", () => {
  it("carries exactly the three fields Python's stats dict carries", () => {
    expect(Object.keys(statsFor(["yes", "no"], 128, 64).options).sort())
      .toEqual(["distinct", "tokens_per_option", "total"]);
  });

  it("counts the options the question defines, not the markers that survived", () => {
    // Python: 40 labels at maxLen 40 leave 2 markers in the sequence, and `total` stays 40 --
    // a report counted from markers would read "2 of 2 distinct" about a question with 40 labels.
    const wide = built(MANY, 40, 400);
    expect(wide.stats.options.total).toBe(40);
    expect(wide.markers.length).toBe(2);
    expect(wide.stats.options).toEqual({ total: 40, distinct: 4, tokens_per_option: 9 });
  });
});

describe("nothing collapses when the head has room", () => {
  it("one span per option, and no cap applied", () => {
    // Python: {"options": 4, "options_distinct": 4, "tokens_per_option": null} at (512, 256),
    // assembled to 186 tokens.
    expect(statsFor(HUE, 512, 256).options)
      .toEqual({ total: 4, distinct: 4, tokens_per_option: null });
    expect(built(HUE, 512, 256).ids.length).toBe(186);
  });
});

describe("the cut collapses shared prefixes", () => {
  it("applies the cap and loses three of the four spans", () => {
    // Python: floor((24 - 16) / 4) = 2, raised to the floor of 4, so every option is cut to
    // " iot" -- one span for four labels.
    const tight = built(HUE, 128, 24);
    expect(tight.stats.options).toEqual({ total: 4, distinct: 1, tokens_per_option: 4 });
    expect(tight.markers.length).toBe(4);
    expect(tight.ids.length).toBe(35);
  });

  it("is the question's number, not the state's", () => {
    const other = statsFor(HUE, 128, 24, "a completely different request");
    expect(other.options.distinct).toBe(statsFor(HUE, 128, 24).options.distinct);
  });

  it("keeps spans that differ, at the same budget", () => {
    // The report is about the option texts: four labels that part after two characters all keep
    // their own span under the same cap.
    expect(statsFor(["alpha", "bravo", "charlie", "delta"], 128, 24).options)
      .toEqual({ total: 4, distinct: 4, tokens_per_option: 4 });
  });
});

describe("collapsedOptions filters as laya.common.collapsed_options does", () => {
  const tight = statsFor(HUE, 128, 24);
  const roomy = statsFor(HUE, 512, 256);

  it("names only the questions that lost a span", () => {
    expect(Object.keys(collapsedOptions(["intent", "dept"], [tight, roomy]))).toEqual(["intent"]);
    expect(collapsedOptions(["intent"], [tight])["intent"])
      .toEqual({ total: 4, distinct: 1, tokens_per_option: 4 });
  });

  it("is empty when nothing collapsed", () => {
    expect(collapsedOptions(["dept"], [roomy])).toEqual({});
  });

  it("skips a state with no stats at all", () => {
    expect(collapsedOptions(["a", "b"], [undefined, roomy])).toEqual({});
  });
});

/** A real Agent whose model is a fake provider: only the reporting is under test. */
function makeAgent() {
  const provider = {
    async runEncoder(b: { inputIds: number[][] }) {
      return { lastHidden: b.inputIds.map(() => [[1, 0], [0, 1]]) };
    },
    async runHead(_h: unknown, b: { inputIds: number[][], markerPos: number[][] }) {
      return {
        logits: b.inputIds.map((_r, i) => b.markerPos[i].map((_m, k) => 4 - k)),
        act: b.inputIds.map(() => [3, 0]),
      };
    },
  };
  return new Agent({ provider, tok: TOK } as never);
}

const QUESTIONS = { intent: { type: "choice", instructions: "which one?",
  criteria: Object.fromEntries(HUE.map((k) => [k, k.replace(/_/g, " ")])) } };
// Long enough that the 128-token window cannot keep it, so the clamp and the cap both bite.
const LONG_STATE = Array.from({ length: 200 }, () => "word").join(" ");

describe("what the Agent publishes", () => {
  it("puts the collapse in usage.options when the head budget forced it", async () => {
    const out = await makeAgent().systemOne("a state", QUESTIONS as never,
      { maxLen: 128, headMaxLen: 24 });
    expect(out.usage.options).toEqual({ intent: { total: 4, distinct: 1, tokens_per_option: 4 } });
  });

  it("says nothing when every option kept its own span", async () => {
    const out = await makeAgent().systemOne("a state", QUESTIONS as never,
      { maxLen: 512, headMaxLen: 256 });
    expect(out.usage.options).toBeUndefined();
    expect(Object.keys(out.usage!).sort()).toEqual(
      ["input_tokens", "output_tokens", "state_tokens", "state_tokens_dropped", "truncated",
        "truncated_questions"]);
  });

  it("reports a cut state and a cut head as the two separate facts they are", async () => {
    const out = await makeAgent().systemOne(LONG_STATE, QUESTIONS as never,
      { maxLen: 128, headMaxLen: 24 });
    expect(out.usage.truncated).toBe(true);
    expect(out.usage.truncated_questions).toEqual(["intent"]);
    expect(out.usage.options).toEqual({ intent: { total: 4, distinct: 1, tokens_per_option: 4 } });
  });
});
