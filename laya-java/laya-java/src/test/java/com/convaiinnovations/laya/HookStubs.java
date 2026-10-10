package com.convaiinnovations.laya;

import com.convaiinnovations.laya.hooks.Hook;
import com.convaiinnovations.laya.hooks.PredictContext;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

/**
 * The hook chain {@code fixtures/hooks.json} was recorded over, rebuilt on this side.
 *
 * <p>The fixture does not record closures, it records a NAME, a set of events and a behaviour
 * string per event — so the same chain can be built in either language from the same data. This
 * is the Java half of that contract.
 *
 * <p>The class is called {@code StubHook} because the reference's stub is, and
 * {@link com.convaiinnovations.laya.hooks.Hooks#dispatch} quotes the hook's class name in the
 * line it reports when a failure is swallowed. The recorded text is part of what the fixture
 * pins, so the name is data rather than taste.
 */
final class HookStubs {

    private HookStubs() {
    }

    /** One result in the shape a prediction has, identified by its input-token count. */
    static Prediction result(int i) {
        return new Prediction(Prediction.MODEL, Map.of(),
                new Usage(10 + i, i, 7, 0, false, List.of(), Map.of()));
    }

    /** The questions a case asks: one noul each, which is all these cases need asked. */
    static Map<String, Question> questions(List<String> ids) {
        Map<String, Question> out = new LinkedHashMap<>();
        for (String id : ids) {
            out.put(id, Question.noul("is this " + id));
        }
        return out;
    }

    /** A hook named {@code name}, implementing {@code events}, doing {@code behaviour}. */
    static StubHook hook(List<Object> log, String name, List<String> events,
                         Map<String, String> behaviour, boolean observe) {
        return new StubHook(log, name, events, behaviour, observe);
    }

    /** A hook that only records, implementing every event. */
    static StubHook hook(List<Object> log, String name) {
        return new StubHook(log, name, Hooks0.EVENTS, Map.of(), false);
    }

    /** The six wire names, as the fixture spells them. */
    static final class Hooks0 {
        static final List<String> EVENTS = List.of("on_predict_start", "on_predict_end",
                "on_route", "on_load", "on_evict", "on_error");

        private Hooks0() {
        }
    }

    /** What one hook saw, keyed exactly as the fixture records it. */
    static Map<String, Object> observation(String name, String event, PredictContext ctx) {
        Map<String, Object> seen = new LinkedHashMap<>();
        seen.put("call", name + "." + event);
        seen.put("states", new ArrayList<>(ctx.states()));
        seen.put("questions", new ArrayList<>(ctx.questions().keySet()));
        seen.put("model", ctx.model());
        seen.put("max_len", ctx.maxLen());
        seen.put("head_max_len", ctx.headMaxLen());
        if (ctx.results() == null) {
            seen.put("results", null);
        } else {
            List<Object> tokens = new ArrayList<>();
            for (Prediction prediction : ctx.results()) {
                tokens.add(prediction.usage().inputTokens());
            }
            seen.put("results", tokens);
        }
        if (ctx.usage() == null) {
            seen.put("usage", null);
        } else {
            seen.put("usage", Map.of("input_tokens", ctx.usage().inputTokens(),
                    "output_tokens", ctx.usage().outputTokens()));
        }
        seen.put("error", ctx.error() == null ? null : ctx.error().getMessage());
        seen.put("elapsed_ms_set", ctx.elapsedMs() != null);
        seen.put("run_id", ctx.runId());
        return seen;
    }

    /**
     * A recorded hook.
     *
     * <p>Every one of the six methods is overridden and then filtered on {@code events}, rather
     * than only overriding the ones a case names: the reference's stub has only the named
     * methods and {@code dispatch} skips what is absent, and the observable it produces — the
     * hook logs nothing for an event it does not implement — is the same either way. This is
     * where the port's "a default no-op is an absent method" claim is actually exercised.
     */
    static final class StubHook implements Hook {

        private final List<Object> log;
        private final String name;
        private final Set<String> events;
        private final Map<String, String> behaviour;
        private final boolean observe;

        private StubHook(List<Object> log, String name, List<String> events,
                         Map<String, String> behaviour, boolean observe) {
            this.log = log;
            this.name = name;
            this.events = new LinkedHashSet<>(events);
            this.behaviour = behaviour;
            this.observe = observe;
        }

        String name() {
            return name;
        }

        @Override
        public void onPredictStart(PredictContext ctx) {
            fire("on_predict_start", ctx);
        }

        @Override
        public void onPredictEnd(PredictContext ctx) {
            fire("on_predict_end", ctx);
        }

        @Override
        public void onRoute(PredictContext ctx) {
            fire("on_route", ctx);
        }

        @Override
        public void onLoad(PredictContext ctx) {
            fire("on_load", ctx);
        }

        @Override
        public void onEvict(PredictContext ctx) {
            fire("on_evict", ctx);
        }

        @Override
        public void onError(PredictContext ctx) {
            fire("on_error", ctx);
        }

        private void fire(String event, PredictContext ctx) {
            if (!events.contains(event)) {
                return;
            }
            log.add(observe ? observation(name, event, ctx) : name + "." + event);
            apply(behaviour.get(event), ctx);
        }

        /** The behaviour language the fixture writes, read back. */
        private void apply(String spec, PredictContext ctx) {
            if (spec == null || "record".equals(spec)) {
                return;
            }
            int colon = spec.indexOf(':');
            String verb = colon < 0 ? spec : spec.substring(0, colon);
            String argument = colon < 0 ? "" : spec.substring(colon + 1);
            switch (verb) {
                case "raise" ->
                        // The reference raises a ValueError; this is its Java counterpart, and
                        // the MESSAGE is what the fixture compares.
                        throw new IllegalArgumentException("boom from " + name);
                case "skip" -> ctx.skip(results(0, Integer.parseInt(argument)));
                case "states" -> ctx.states(split(argument));
                case "questions" -> ctx.questions(questions(split(argument)));
                case "results" -> ctx.results(results(500, Integer.parseInt(argument)));
                case "max_len" -> ctx.maxLen(Integer.parseInt(argument));
                case "head_max_len" -> ctx.headMaxLen(Integer.parseInt(argument));
                case "sleep" -> sleep(Double.parseDouble(argument));
                default -> throw new AssertionError("unknown hook behaviour " + spec);
            }
        }

        private static void sleep(double seconds) {
            try {
                Thread.sleep((long) (seconds * 1000.0));
            } catch (InterruptedException interrupted) {
                Thread.currentThread().interrupt();
            }
        }

        private static List<String> split(String csv) {
            return csv.isEmpty() ? List.of() : List.of(csv.split(","));
        }

        private static List<Prediction> results(int base, int count) {
            List<Prediction> out = new ArrayList<>(count);
            for (int i = 0; i < count; i++) {
                out.add(result(base + i));
            }
            return out;
        }

        @Override
        public String toString() {
            return name;
        }
    }
}
