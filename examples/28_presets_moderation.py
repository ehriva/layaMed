"""Example 28 -- moderation preset across four kinds of post.

Runs `laya.moderation_questions()` over a targeted insult, a generic insult, spam and a
benign message, so each category's signal pattern is visible. The summary at the bottom is
computed from the probabilities this run measured, because a conclusion typed into a print
statement describes one model's output and silently goes stale on the next.
"""

from _common import laya, banner, describe, heading, load

banner("28", "Preset: content moderation", """
    `moderation_questions()` asks the four safety questions a moderation queue cares about
    -- toxic, harassment, threat, spam -- plus an ordinal severity rubric, all in one
    forward pass. The point of the example is the shape of each pattern: which questions a
    post carries past 0.5. A targeted insult lights up harassment as well as toxic, a
    generic one lights up toxic alone, spam lights up spam alone, and an ordinary post
    lights up nothing.

    Nothing here means nothing past 0.5, not zero: an ordinary post still reads a little
    on `spam`, and which question crosses the line is a property of this checkpoint on this
    device. So every name, gap and ranking below is read off the probabilities printed above
    instead of being hardcoded -- `tests/test_confidence.py` runs the functions on fabricated
    answers and fails if a conclusion stops following its data.
    """)

POSTS = [
    ("targeted insult", "Nobody asked for your opinion. You are the most useless person on "
                        "this forum and everyone knows it."),
    ("generic insult", "Honestly this is the dumbest post I have read all week. "
                       "Did you even think before typing?"),
    ("spam", "MAKE $5000 A WEEK FROM HOME! Click here now: bit.ly/xyz  "
             "Limited spots, DM me to start today!"),
    ("benign", "I switched to the new scheduler and my build times dropped by about 30 percent. "
               "Happy to share the config."),
]


# --- the page's conclusions, as pure functions over answers so a test can drive them ---------
def noul_flags(questions):
    """The preset's flag questions, in declaration order -- derived, so a fifth flag shows up."""
    return [k for k, q in questions.items() if q["type"] == "noul"]


def score_ceiling(questions):
    """The top of the ordinal rubric: the last criterion's index, not a typed-in 3."""
    return max(len(q["criteria"]) - 1 for q in questions.values() if q["type"] == "score")


def flag_line(answers, flags):
    """Every flag with its probability. A line that speaks about all of them prints all of them."""
    return "  ".join("`%s` %.3f" % (k, answers[k]["noul"]) for k in flags)


def past_half(answers, flags):
    """The flags this answer reads as true -- the same 0.5 cut `_common.describe` labels."""
    return [k for k in flags if answers[k]["noul"] > 0.5]


def gaps(a, b, flags):
    """Per-flag difference between two answers, widest first; `+` means `a` reads higher."""
    g = [(k, a[k]["noul"] - b[k]["noul"]) for k in flags]
    g.sort(key=lambda kv: -abs(kv[1]))
    return g


def rank_by_severity(results, labels):
    """Rank the posts on `severity`, widest score first.

    Returns (chain, in_listed_order). The chain's `>` signs are only ever printed between
    scores that really descend, and the verdict compares the ranking to the order the posts
    are listed in -- so neither can be asserted by the format string that prints it.
    """
    ranked = sorted(labels, key=lambda l: -results[l]["severity"]["score"])
    chain = " > ".join("%s %.2f" % (l, results[l]["severity"]["score"]) for l in ranked)
    return chain, ranked == list(labels)


agent = load("english")

QUESTIONS = laya.moderation_questions()
FLAGS = noul_flags(QUESTIONS)
CEILING = score_ceiling(QUESTIONS)

results = {}
for label, post in POSTS:
    heading(label)
    answers = agent.predict({"post": post}, QUESTIONS)["answers"]
    results[label] = answers
    describe(answers)

heading("side by side")
print("   %-16s %s  severity" % ("post", "  ".join("%-12s" % k for k in FLAGS)))
for label, _ in POSTS:
    a = results[label]
    print("   %-16s %s  %.2f / %d"
          % (label, "  ".join("%-12.3f" % a[k]["noul"] for k in FLAGS),
             a["severity"]["score"], CEILING))

heading("what the pattern means")
for label, _ in POSTS:
    a = results[label]
    hits = past_half(a, FLAGS)
    print("   %-16s -> %s" % (label, flag_line(a, FLAGS)))
    print("   %-16s    past 0.5: %s"
          % ("", ", ".join("`%s`" % k for k in hits) or "nothing"))

tgt, gen = results["targeted insult"], results["generic insult"]
g = gaps(tgt, gen, FLAGS)
print("\n   the two hostile posts, per-flag gap (widest first, `+` = the targeted one reads higher):")
print("   %s" % ", ".join("`%s` %+.3f" % (k, v) for k, v in g))
print("   the question that separates them most is `%s` (%.3f) and the least is `%s` (%.3f),"
      % (g[0][0], abs(g[0][1]), g[-1][0], abs(g[-1][1])))
print("   so the least of them cannot tell a targeted insult from a generic one on its own.")

chain, in_order = rank_by_severity(results, [label for label, _ in POSTS])
print("\n   `severity` ranks the set: %s" % chain)
print("   that is%s the order the posts are listed in. Whatever the order, `severity` is a coarse"
      % ("" if in_order else " not"))
print("   0-%d rubric, so use it to sort a queue rather than as a hard threshold." % CEILING)
