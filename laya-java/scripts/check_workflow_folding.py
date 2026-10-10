#!/usr/bin/env python3
"""Refuse a workflow `run:` block that folds a comment into the command.

A folded scalar -- `run: >` or `run: >-` -- joins its lines with spaces, and `#` is not a comment
inside a YAML scalar. So a comment written inside one arrives at the shell as part of the command
and silences everything after it on the folded line. A literal block -- `run: |` -- keeps the
newlines, so there the `#` really is a shell comment. The two forms are one character apart and
look identical in a diff.

This is not hypothetical. `release-java.yml`'s release gate read:

    - name: The suite was not vacuous
      run: >
        python3 scripts/check_test_results.py laya-java/build/test-results/test
        # 631 of the 640 registered tests run without a checkpoint. ...
        --min-tests 625 --allow-aborted

which the shell received as

    python3 scripts/check_test_results.py laya-java/build/test-results/test

with both arguments gone: the floor fell to the script's default of 1 instead of rising to 625,
and `--allow-aborted` was lost on a runner whose parity factories abort. One failure is silent and
one is loud, and the step's own purpose was to stop a gate that does not bind. `actionlint` does
not report it, because the YAML is valid and the shell command is a legal one.

Exits 1 and names every occurrence. Run with no arguments to scan `.github/workflows/*.yml` from
the repository root, or pass paths.
"""
import glob
import io
import os
import re
import sys

# `run:` introducing a block scalar, with the chomping indicator optional.
RUN_BLOCK = re.compile(r'^(\s*)run:\s*(>-?|\|-?)\s*$')


def offenders(path):
    """Every (line number, comment) a folded `run:` block in `path` would hand to the shell."""
    lines = io.open(path, encoding="utf-8").read().split("\n")
    found = []
    index = 0
    while index < len(lines):
        match = RUN_BLOCK.match(lines[index])
        if match is None:
            index += 1
            continue
        indent, style = match.group(1), match.group(2)
        start = index
        index += 1
        body = []
        while index < len(lines):
            line = lines[index]
            # The block ends at the first non-blank line indented no further than `run:` itself.
            if line.strip() and len(line) - len(line.lstrip()) <= len(indent):
                break
            body.append(line)
            index += 1
        if not style.startswith(">"):
            continue                      # a literal block keeps its newlines; `#` is fine there
        for line in body:
            if line.strip().startswith("#"):
                found.append((start + 1, line.strip()))
    return found


def main(argv):
    paths = argv[1:]
    if not paths:
        root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
        paths = sorted(glob.glob(os.path.join(root, ".github", "workflows", "*.yml")))
    if not paths:
        print("check_workflow_folding: no workflows found", file=sys.stderr)
        return 1

    total = 0
    for path in paths:
        for line, comment in offenders(path):
            total += 1
            print("%s:%d: a comment inside a folded `run: >` block reaches the shell and"
                  " truncates the command" % (os.path.relpath(path), line), file=sys.stderr)
            print("    %s" % comment[:110], file=sys.stderr)
    if total:
        print("check_workflow_folding: %d folded block(s) with an embedded comment. Move the"
              " comment above the `run:` key, or use `run: |`." % total, file=sys.stderr)
        return 1
    print("check_workflow_folding: %d workflow(s) clean" % len(paths))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
