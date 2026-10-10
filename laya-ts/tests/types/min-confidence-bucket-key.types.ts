/**
 * Compile-time regression guard for the `min_confidence` bucket-key vocabulary (#1002 TypeScript
 * parity). `tsc -p tests/types/tsconfig.json` type-checks this file: a key the runtime cannot
 * produce has to be rejected by the type, not only by `checkMinConfidenceMap` at run time.
 */
import type { MinConfidenceMap } from "../../src/common.js";

// Every bucket `optionBucket` can produce, plus `default`, keeps type-checking.
export const valid: MinConfidenceMap = {
  "choice:2": 0.9,
  "choice:3-5": 0.8,
  "choice:6-10": 0.7,
  "choice:11+": 0.6,
  "score:3-5": 0.5,
  "noul:2": 0.4,
  default: 0.3,
};

// A map assembled at run time is still accepted: `Record<string, number>` stays assignable.
export const dynamic: MinConfidenceMap = Object.fromEntries([["choice:2", 0.4]]);

// An impossible size reads as a bucket but cannot be produced, so it is a type error.
// @ts-expect-error "choice:2-5" is not a bucket the runtime can produce
export const impossibleSize: MinConfidenceMap = { "choice:2-5": 0.9 };
// @ts-expect-error "foo" is not an option type
export const unknownType: MinConfidenceMap = { "foo:2": 0.9 };
// @ts-expect-error "score:1" is not a bucket the runtime can produce
export const impossibleScoreSize: MinConfidenceMap = { "score:1": 0.9 };
// @ts-expect-error values are thresholds, not strings
export const badValue: MinConfidenceMap = { "choice:2": "0.9" };
