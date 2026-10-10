/** The option types and option-count sizes core's `temp_bucket` can produce. The bucket-key type
 *  and the runtime `BUCKET_KEY` check below are both derived from these, so the keys a
 *  `min_confidence` map may name cannot drift from the ones the engine can answer with. */
const BUCKET_TYPES = ['choice', 'score', 'noul'] as const;
const BUCKET_SIZES = ['2', '3-5', '6-10', '11+'] as const;

/** One bucket key core can produce, e.g. `"choice:2"` or `"score:11+"`. */
export type MinConfidenceBucket = `${(typeof BUCKET_TYPES)[number]}:${(typeof BUCKET_SIZES)[number]}`;
/** A `MinConfidenceMap` key: a bucket core can produce, or `"default"` for the rest. */
export type MinConfidenceKey = MinConfidenceBucket | 'default';

const BUCKET_KEY = new RegExp(
  `^(${BUCKET_TYPES.join('|')}):(${BUCKET_SIZES.map(size => size.replace('+', '\\+')).join('|')})$`,
);

/** True for a key core's `temp_bucket` can produce, or `"default"`. Mirrors core's
 *  `check_min_confidence`, so a gate the server would 422 is refused before the request. */
export function isMinConfidenceKey(key: string): key is MinConfidenceKey {
  return key === 'default' || BUCKET_KEY.test(key);
}
