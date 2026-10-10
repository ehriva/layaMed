#!/usr/bin/env python3
"""Fail when any Java dependency this module ships has a known advisory in OSV.

`security.yml` audits the Python dependencies with pip-audit and nothing audited the Java tree at
all, which is the wrong way round for the one artifact in this repository that gets published to
Maven Central: a consumer who adds `com.convaiinnovations:laya-java` inherits its compile and
runtime dependencies, and a version burned on Central cannot be withdrawn.

Reads Gradle's dependency-tree output on stdin and extracts the resolved coordinates, so
TRANSITIVE dependencies are covered rather than only the ones the build file names:

    ./gradlew -q :laya-java:dependencies --configuration runtimeClasspath \\
        | python3 laya-java/scripts/check_java_dependencies.py

Queries https://api.osv.dev, which aggregates GitHub Security Advisories among others and needs
no account or token -- a gate that requires a secret is a gate that silently stops running on a
fork. Exits 1 on any advisory, 0 when the tree is clean, and 2 when the scan itself could not run
(no coordinates found, or OSV unreachable), because "the scanner failed" must not read as "the
dependencies are clean".
"""
import io
import json
import re
import shutil
import subprocess
import sys
import urllib.error
import urllib.request

OSV = "https://api.osv.dev/v1/querybatch"

# A Gradle tree line, e.g. "+--- com.microsoft.onnxruntime:onnxruntime:1.20.0" or
# "|    \\--- org.foo:bar:1.2 -> 1.3" (the arrow means the resolved version differs from the
# requested one, and the RESOLVED version is the one that ships).
COORDINATE = re.compile(
    r"^[|\\+\- ]*"
    r"(?P<group>[A-Za-z0-9_.\-]+):(?P<name>[A-Za-z0-9_.\-]+):"
    r"(?P<requested>[A-Za-z0-9_.\-+]+)"
    r"(?:\s*->\s*(?P<resolved>[A-Za-z0-9_.\-+]+))?")


def coordinates(text):
    """Every distinct group:name:version in a Gradle dependency tree, resolved versions."""
    found = {}
    for line in text.split("\n"):
        if "(n)" in line or "FAILED" in line:
            # `(n)` marks a configuration that was not resolved, so it carries no real version.
            continue
        match = COORDINATE.match(line.rstrip())
        if match is None:
            continue
        version = match.group("resolved") or match.group("requested")
        name = "%s:%s" % (match.group("group"), match.group("name"))
        found[(name, version)] = True
    return sorted(found)


def query(pairs):
    """OSV's batch endpoint, one query per coordinate, in the same order.

    Tries `urllib` first and falls back to `curl`. Not belt-and-braces: a sandboxed developer
    environment can permit `curl` while Python's own socket layer cannot reach anything --
    measured here, where `urllib` failed against both api.osv.dev and api.github.com while curl
    returned 200 for the same request in under a second. A gate that depends on one HTTP client
    stops running for reasons that have nothing to do with the dependencies it audits, and a gate
    that stops running is the thing this file exists to prevent.
    """
    payload = json.dumps({"queries": [
        {"version": version, "package": {"name": name, "ecosystem": "Maven"}}
        for name, version in pairs]}).encode("utf-8")

    try:
        request = urllib.request.Request(
            OSV, data=payload, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(request, timeout=60) as response:
            return json.loads(response.read().decode("utf-8"))
    except (urllib.error.URLError, OSError, ValueError) as urllib_problem:
        curl = shutil.which("curl")
        if curl is None:
            raise
        done = subprocess.run(
            [curl, "-sS", "--max-time", "60", "-X", "POST", OSV,
             "-H", "Content-Type: application/json", "--data-binary", "@-"],
            input=payload, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
        if done.returncode != 0:
            raise OSError("urllib failed (%s) and curl failed (%s)"
                          % (urllib_problem, done.stderr.decode("utf-8", "replace").strip()))
        print("  (urllib could not reach OSV here, so this used curl: %s)" % urllib_problem)
        return json.loads(done.stdout.decode("utf-8"))


def main(argv):
    text = io.open(argv[1], encoding="utf-8").read() if len(argv) > 1 else sys.stdin.read()
    pairs = coordinates(text)
    if not pairs:
        print("check_java_dependencies: no coordinates in the input; the dependency tree did not"
              " reach this script, so nothing was audited", file=sys.stderr)
        return 2

    try:
        answer = query(pairs)
    except (urllib.error.URLError, OSError, ValueError) as problem:
        print("check_java_dependencies: could not reach OSV (%s); refusing to report a clean"
              " tree on a scan that did not run" % problem, file=sys.stderr)
        return 2

    results = answer.get("results") or []
    if len(results) != len(pairs):
        print("check_java_dependencies: OSV answered %d of %d queries; refusing to call the"
              " remainder clean" % (len(results), len(pairs)), file=sys.stderr)
        return 2

    total = 0
    for (name, version), result in zip(pairs, results):
        vulns = result.get("vulns") or []
        if vulns:
            total += len(vulns)
            for vuln in vulns:
                print("%s:%s  %s" % (name, version, vuln.get("id")), file=sys.stderr)
        else:
            print("  %s:%s clean" % (name, version))
    if total:
        print("check_java_dependencies: %d advisory/advisories across %d dependency/dependencies."
              " See https://osv.dev for each id." % (total, len(pairs)), file=sys.stderr)
        return 1
    print("check_java_dependencies: %d dependency/dependencies, no known advisories"
          % len(pairs))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
