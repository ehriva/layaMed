#!/usr/bin/env bash
# Every scanner the pipeline runs against laya-java, locally, in one command.
#
# This exists because "the scanner will catch it in CI" means shipping a defect to a reviewer.
# CodeQL found an unreleased lock in `Router.build` on its first CI run -- a `finally` that called
# `finish(...)` before `lock.unlock()`, so a throw from `finish` left the lock held with no owner
# able to release it. Semgrep could not have found it: an unreleased lock is a control-flow
# property, and Semgrep matches patterns. The lesson is not "add Semgrep rules", it is "run the
# engine that reasons about control flow before pushing, not after".
#
# Usage, from the repository root:
#
#     laya-java/scripts/scan_local.sh                 # everything that needs no download
#     CODEQL=/path/to/codeql laya-java/scripts/scan_local.sh    # ...and CodeQL too
#
# A CodeQL CLI is a ~700 MB download (the `codeql-bundle-*` asset on github/codeql-action
# releases, which carries the query packs). Without $CODEQL this script SKIPS it and says so
# loudly, and exits non-zero, because a partial scan reported as a pass is the thing that let the
# lock through.
set -u
cd "$(dirname "$0")/../.." || exit 2

status=0
note() { printf '\n=== %s ===\n' "$1"; }
fail() { printf '  FAILED: %s\n' "$1"; status=1; }

note "workflow folding (a comment in a run: > block truncates the command)"
python3 laya-java/scripts/check_workflow_folding.py || fail "check_workflow_folding"

note "semgrep (p/java + p/security-audit, --error so findings block)"
if command -v semgrep > /dev/null 2>&1; then
    semgrep --config p/java --config p/security-audit --error --metrics=off \
        laya-java/laya-java/src/main/java || fail "semgrep"
else
    fail "semgrep is not installed (pip install semgrep)"
fi

note "java dependency advisories (OSV)"
( cd laya-java && ./gradlew --no-daemon -q :laya-java:dependencies \
    --configuration runtimeClasspath ) \
    | python3 laya-java/scripts/check_java_dependencies.py || fail "check_java_dependencies"

note "codeql (java-kotlin, security-extended)"
if [ -n "${CODEQL:-}" ] && [ -x "${CODEQL}" ]; then
    db="${TMPDIR:-/tmp}/laya-java-codeql-db"
    rm -rf "$db"
    # `--command` rather than autobuild: autobuild would pick its own Gradle invocation, and this
    # has to be the same compile the module ships -- release 17, -Werror.
    if "$CODEQL" database create "$db" --language=java-kotlin --overwrite \
            --source-root=. \
            --command="laya-java/gradlew --no-daemon -p laya-java :laya-java:compileJava" \
            > /dev/null; then
        "$CODEQL" database analyze "$db" \
            codeql/java-queries:codeql-suites/java-security-extended.qls \
            --format=sarif-latest --output="${TMPDIR:-/tmp}/laya-java.sarif" > /dev/null \
            || fail "codeql analyze"
        python3 - "${TMPDIR:-/tmp}/laya-java.sarif" <<'PY' || fail "codeql findings"
import io, json, sys
sarif = json.load(io.open(sys.argv[1], encoding="utf-8"))
results = [r for run in sarif.get("runs", []) for r in run.get("results", [])]
if not results:
    print("  codeql: no findings")
    raise SystemExit(0)
for r in results:
    loc = (r.get("locations") or [{}])[0].get("physicalLocation", {})
    print("  %s  %s:%s  %s" % (
        r.get("ruleId"),
        loc.get("artifactLocation", {}).get("uri"),
        loc.get("region", {}).get("startLine"),
        (r.get("message") or {}).get("text", "")[:90]))
print("  codeql: %d finding(s)" % len(results))
raise SystemExit(1)
PY
    else
        fail "codeql database create"
    fi
else
    printf '  SKIPPED: set CODEQL to a codeql CLI to run it.\n'
    printf '  Not optional before a push: CodeQL is the only one of these that reasons about\n'
    printf '  control flow, and it is what found the unreleased lock the others missed.\n'
    status=1
fi

note "result"
if [ "$status" -eq 0 ]; then
    echo "  every scanner ran and found nothing"
else
    echo "  at least one scanner failed, was skipped, or reported findings -- do not push"
fi
exit "$status"
