import { describe, expect, it, vi } from "vitest";
import { Router, normaliseName } from "../src/router.js";
describe("router", () => {
  it("normalises aliases", () => { expect(normaliseName("en")).toBe("english"); });
  it("rejects unknown model", () => { expect(() => normaliseName("jev-1")).toThrow(); });
  it.each(["refund", "12345 ???", { body: "refund" }])("uses multilingual for unidentified state %j", (state) => {
    const r = new Router();
    const decision = r.route(state, {});
    expect(r.default).toBe("multilingual");
    expect(decision.detection?.languageUndecided).toBe(true);
    expect(decision.model).toBe("multilingual");
    expect(decision.reason).toContain("using default (multilingual)");
    expect(r.loaded).toEqual([]);
  });
  it.each(["english", "multilingual"])("applies default %s only to unidentified text", (fallback) => {
    const r = new Router({ default: fallback });
    expect(r.route("refund", {}).model).toBe(fallback);
    expect(r.route("Please refund the duplicate charge on the invoice", {}).model).toBe("english");
    expect(r.route("मुझसे दो बार शुल्क लिया गया", {}).model).toBe("multilingual");
    expect(r.route("refund", {}, { lang: "en" }).model).toBe("english");
    expect(r.route("refund", {}, { model: "typed-decisions" }).model).toBe("typed-decisions");
  });
  it("uses the default in a mixed routing batch", () => {
    const r = new Router();
    const decisions = r.routeBatch([
      { state: "refund", questions: {} },
      { state: "Please refund the duplicate charge on the invoice", questions: {} },
    ]);
    expect(decisions.map((d) => d.model)).toEqual(["multilingual", "english"]);
    expect(r.loaded).toEqual([]);
  });
  it("loads the multilingual checkpoint when predicting unidentified text", async () => {
    const loader = vi.fn(async () => ({
      async systemOne() {
        return { model: "laya-rl-agent", answers: {}, usage: { input_tokens: 5, output_tokens: 0 } };
      },
    }));
    const r = new Router({ loader });
    const result = await r.predict("refund", { q: { type: "noul", instructions: "?" } });
    expect(loader).toHaveBeenCalledWith("multilingual", r.models.multilingual);
    expect(result.routing.model).toBe("multilingual");
    expect(r.loaded).toEqual(["multilingual"]);
  });
  it("routes hindi script to multilingual without loading", () => {
    const r = new Router();
    expect(r.route({ body: "मुझसे दो बार शुल्क लिया गया" }, {}).model).toBe("multilingual");
    expect(r.loaded).toEqual([]);
  });
  it.each([
    ["Please send me the café menu today please", "english"],
    ["Send the invoice to José before Friday", "english"],
    ["Grüße aus Köln, wir melden uns wegen der Rechnung", "multilingual"],
    ["sluk lyset i soveværelset", "multilingual"],
  ])("routes %s to %s", (text, model) => {
    expect(new Router().route(text, {}).model).toBe(model);
  });
  it("explicit model wins", () => {
    expect(new Router().route("hi", {}, { model: "typed-decisions" }).model).toBe("typed-decisions");
  });
  it("lang_guess callable routes", () => {
    const r = new Router();
    expect(r.route("text", {}, { langGuess: () => "ro" }).model).toBe("multilingual");
  });
  it("shares concurrent loads and emits lifecycle hooks once per cached model", async () => {
    const events: string[] = [];
    const loader = vi.fn(async (name: string) => ({ name }));
    const r = new Router({
      maxLoaded: 1,
      loader,
      hooks: [{
        onLoad: (ctx) => events.push(`load:${ctx.model}`),
        onEvict: (ctx) => events.push(`evict:${ctx.model}`),
      }],
    });

    const [first, second] = await Promise.all([r.load("en"), r.load("english")]);
    expect(first).toBe(second);
    expect(loader).toHaveBeenCalledTimes(1);
    expect(r.loaded).toEqual(["english"]);
    expect(r._agents.size).toBe(1);
    await r.load("multilingual");
    expect(events).toEqual(["load:english", "evict:english", "load:multilingual"]);
    expect(r.loaded).toEqual(["multilingual"]);
  });
  it("clears an unsuccessful in-flight load so the model can be retried", async () => {
    const agent = { name: "english" };
    const loader = vi.fn()
      .mockRejectedValueOnce(new Error("load failed"))
      .mockResolvedValue(agent);
    const r = new Router({ loader });

    const results = await Promise.allSettled([r.load("en"), r.load("english")]);
    expect(results.map((result) => result.status)).toEqual(["rejected", "rejected"]);
    expect(loader).toHaveBeenCalledTimes(1);
    expect(await r.load("english")).toBe(agent);
    expect(loader).toHaveBeenCalledTimes(2);
    expect(r.loaded).toEqual(["english"]);
  });
  it("rejects invalid inputs in predict before loading any checkpoint", async () => {
    const loader = vi.fn();
    const r = new Router({ loader });

    await expect(r.predict(null, { q: { type: "noul", instructions: "?" } })).rejects.toThrow(
      TypeError,
    );
    await expect(r.predict(null, { q: { type: "noul", instructions: "?" } })).rejects.toThrow(
      /state must not be None/,
    );

    await expect(r.predict("text", [] as never)).rejects.toThrow(TypeError);
    await expect(r.predict("text", [] as never)).rejects.toThrow(
      /questions must be a dict of question id -> definition/,
    );

    await expect(r.predict("text", null as never)).rejects.toThrow(TypeError);
    await expect(r.predict("text", null as never)).rejects.toThrow(
      /questions must be a dict of question id -> definition/,
    );

    expect(loader).not.toHaveBeenCalled();
    expect(r.loaded).toEqual([]);
  });
  it("rejects invalid questions type in route", () => {
    const r = new Router();
    expect(() => r.route("text", [] as never)).toThrow(TypeError);
    expect(() => r.route("text", [] as never)).toThrow(
      /questions must be a dict of question id -> definition/,
    );
  });
  it("dispatches onError and onPredictEnd when load fails inside predict lifecycle", async () => {
    const loader = vi.fn(async () => {
      throw new Error("checkpoint fetch failed");
    });
    const r = new Router({ loader });
    const events: string[] = [];

    await expect(
      r.predict("text", { q: { type: "noul", instructions: "?" } }, {
        hooks: [
          {
            onError(ctx) {
              events.push(`error:${(ctx.error as Error).message}`);
            },
            onPredictEnd(ctx) {
              events.push(`end:${(ctx.error as Error).message}`);
            },
          },
        ],
      }),
    ).rejects.toThrow("checkpoint fetch failed");

    expect(events).toEqual(["error:checkpoint fetch failed", "end:checkpoint fetch failed"]);
  });
  it("validates question definitions before calling systemOne", async () => {
    const customAgent = {
      systemOne: vi.fn(async () => ({ model: "custom", answers: {}, usage: {} })),
    };
    const r = new Router({ loader: vi.fn(async () => customAgent) });

    await expect(
      r.predict("text", { q: { type: "bogus", instructions: "?" } } as never),
    ).rejects.toThrow(/unknown type/);

    await expect(
      r.predict("text", { "": { type: "noul", instructions: "?" } } as never),
    ).rejects.toThrow(/question id must be a non-empty string/);

    await expect(
      r.predict("text", { q: { type: "noul", instructions: "" } } as never),
    ).rejects.toThrow(/instructions.*must not be empty/);

    expect(customAgent.systemOne).not.toHaveBeenCalled();
  });
  it("allows onPredictStart hook to complete an incomplete question or skip", async () => {
    const customAgent = {
      systemOne: vi.fn(async () => ({ model: "custom", answers: {}, usage: {} })),
    };
    const r = new Router({ loader: vi.fn(async () => customAgent) });

    const res = await r.predict("text", { q: { type: "noul" } as never }, {
      onPredictStart: (ctx) => {
        ctx.questions = { q: { type: "noul", instructions: "Added by hook?" } };
      },
    });
    expect(res.routing).toBeDefined();
    expect(customAgent.systemOne).toHaveBeenCalled();

    const skipped = await r.predict("text", { q: { type: "noul" } as never }, {
      onPredictStart: (ctx) => {
        ctx.skip([{ model: "cached", answers: {}, usage: {} }]);
      },
    });
    expect(skipped.routing).toBeDefined();
    expect(skipped.model).toBe("cached");
  });
});
