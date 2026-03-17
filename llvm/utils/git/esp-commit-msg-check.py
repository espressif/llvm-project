#!/usr/bin/env python3
#
# ======- esp-commit-msg-check - Check commit message format over a range --*- python -*--==#
#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# ==--------------------------------------------------------------------------------------==#
#
# Iterates over a specified range of git commits and checks that the first line of each
# commit message follows the required format:
#
# 1. The message must start with either:
#    - One or more tags in square brackets (e.g. [LLVM], [Clang][Driver]). No spaces are
#      allowed between words and brackets. The bracket list must be followed by a single
#      space.
#    - Or a special prefix: esp/ci:, esp/maint:, or esp/release:
#
# 2. The first word after the prefix must start with a capital letter.
#

import argparse
import re
import subprocess
import sys


def run_git(cmd, cwd=None):
    """Run a git command and return stdout as string, or None on failure."""
    try:
        result = subprocess.run(
            ["git"] + cmd,
            cwd=cwd,
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout
    except subprocess.CalledProcessError as e:
        sys.stderr.write(f"git {' '.join(cmd)} failed: {e.stderr}")
        return None
    except FileNotFoundError:
        sys.stderr.write("error: cannot find git command\n")
        sys.exit(2)


def get_commits_in_range(revision_range):
    """Return list of commit hashes in the given range (oldest first)."""
    stdout = run_git(["rev-list", "--reverse", revision_range])
    if stdout is None:
        return None
    return [sha.strip() for sha in stdout.splitlines() if sha.strip()]


def get_commit_subject(sha, repo_path=None):
    """Return the first line (subject) of the commit message for the given SHA."""
    stdout = run_git(["log", "-1", "--format=%s", sha], cwd=repo_path)
    return stdout.strip() if stdout else None


def check_subject_format(subject):
    """
    Check that the subject line follows the required format.
    Returns (is_valid, error_message).
    """
    if not subject or not subject.strip():
        return False, "Empty subject line"

    line = subject.strip()

    # Special prefixes: esp/ci:, esp/maint:, esp/release:
    esp_prefix = re.match(r"^(esp/(?:ci|maint|release):\s+)", line)
    if esp_prefix:
        rest = line[esp_prefix.end() :].lstrip()
        if not rest:
            return False, "Nothing after esp/ prefix"
        if not rest[0].isupper():
            return False, "First word after prefix must start with capital letter"
        return True, None

    # Bracket prefix: [Word][Word] or [Word], no spaces between words and brackets.
    # Words list is followed by a single space.
    bracket_prefix = re.match(r"^(\[[^\s\]]+\])+\s+", line)
    if bracket_prefix:
        rest = line[bracket_prefix.end() :].lstrip()
        if not rest:
            return False, "Nothing after bracket prefix"
        if not rest[0].isupper():
            return (
                False,
                "First word after bracket prefix must start with capital letter",
            )
        return True, None

    return (
        False,
        "Subject must start with [Tag] or esp/ci:, esp/maint:, or esp/release:",
    )


def main():
    parser = argparse.ArgumentParser(
        description="Check commit message format for commits in a git range.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s HEAD~5..HEAD
  %(prog)s main..feature-branch
  %(prog)s abc123..def456
        """,
    )
    parser.add_argument(
        "revision_range",
        metavar="RANGE",
        help="Git revision range to check (e.g. HEAD~10..HEAD or main..branch)",
    )
    parser.add_argument(
        "-C",
        "--git-dir",
        metavar="PATH",
        dest="repo_path",
        help="Path to the git repository (default: current directory)",
    )
    parser.add_argument(
        "-q",
        "--quiet",
        action="store_true",
        help="Only print commits that fail the check",
    )
    args = parser.parse_args()

    commits = get_commits_in_range(args.revision_range)
    if commits is None:
        sys.exit(2)
    if not commits:
        sys.stderr.write("No commits in range.\n")
        sys.exit(0)

    failed = []
    for sha in commits:
        subject = get_commit_subject(sha, args.repo_path)
        if subject is None:
            failed.append((sha, "Could not read commit message"))
            continue
        valid, err = check_subject_format(subject)
        if not valid:
            failed.append((sha, subject, err))

    if failed:
        for item in failed:
            sha, err = item[0], item[-1]
            subject_line = item[1] if len(item) == 3 else ""
            short_sha = sha[:12] if len(sha) >= 12 else sha
            if args.quiet:
                sys.stderr.write(f"{short_sha}: {err}\n")
            else:
                sys.stderr.write(f"{short_sha} {subject_line}\n  -> {err}\n")
        sys.exit(1)

    if not args.quiet:
        print(f"Checked {len(commits)} commit(s), all passed.")
    sys.exit(0)


if __name__ == "__main__":
    main()
