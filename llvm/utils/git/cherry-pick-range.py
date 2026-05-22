#!/usr/bin/env python3
"""Cherry-pick commits from SOURCE_BRANCH (tag..HEAD, oldest first) onto DEST_BRANCH.

After each successful cherry-pick, the sync tag is force-moved to that source commit.

On cherry-pick failure: writes the failed source commit hash to .esp_rebase_HEAD
at the repository root (overwrites if present). By default runs git cherry-pick --abort,
then exits with status 1 if no commit was applied yet, or 2 if at least one commit was
cherry-picked before the failure. Use --no-abort to leave the failed cherry-pick
in place for manual resolution.

With -c/--continue: runs git cherry-pick --continue first (same failure handling);
on success reads .esp_rebase_HEAD and moves the sync tag to that hash, then proceeds
with the usual cherry-pick range.

With --no-abort and -b/--wip-br: on a cherry-pick conflict, stages all files, runs
git cherry-pick --continue, and creates local branch
sync_fail/<dest_BASE>_<sync_tag_TARGET>_<failed_COMMIT> (7-char short hashes: dest head and
sync-tag target as of script start, and the failed source commit) at HEAD without checking
it out; the branch name is written to .esp_rebase_WIP_BRANCH (overwrites if present). The
current branch is then reset with `git reset --hard HEAD~1` so the conflict cherry-pick commit
is dropped from the destination while it remains on the WIP branch. The sync tag is not
advanced and remaining picks are skipped.

Requires: pip install GitPython
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path
from typing import Any, NoReturn

try:
    import git
    from git.exc import GitCommandError, InvalidGitRepositoryError
except ImportError:
    git = None  # type: ignore[assignment, misc]
    GitCommandError = Exception  # type: ignore[assignment, misc]
    InvalidGitRepositoryError = Exception  # type: ignore[assignment, misc]

DEFAULT_TAG = "esp_main_synced"
ESP_REBASE_HEAD_FILE = ".esp_rebase_HEAD"
ESP_REBASE_WIP_BRANCH_FILE = ".esp_rebase_WIP_BRANCH"


def _die(msg: str, code: int = 1) -> None:
    print(f"error: {msg}", file=sys.stderr)
    raise SystemExit(code)


def _rev_parse_verify(repo: Any, ref: str) -> str:
    try:
        return repo.git.rev_parse("--verify", ref).strip()
    except GitCommandError as e:
        _die(f"invalid ref {ref!r}: {e}")


def _tag_commit_sha(repo: Any, tag: str) -> str:
    ref = f"{tag}^{{commit}}"
    try:
        return repo.git.rev_parse("--verify", ref).strip()
    except GitCommandError:
        _die(f"tag {tag!r} does not exist (try: git fetch --tags)")


def _commits_between(repo: Any, tag_commit: str, source_ref: str) -> list[str]:
    out = repo.git.rev_list("--reverse", f"{tag_commit}..{source_ref}")
    return [h.strip() for h in out.splitlines() if h.strip()]


def _cherry_pick_in_progress(repo: Any) -> bool:
    try:
        repo.git.rev_parse("--verify", "CHERRY_PICK_HEAD")
        return True
    except GitCommandError:
        return False


def _failed_pick_commit_ref(repo: Any) -> str | None:
    try:
        return repo.git.rev_parse("--verify", "CHERRY_PICK_HEAD").strip()
    except GitCommandError:
        return None


def _current_branch_label(repo: Any) -> str:
    try:
        return repo.git.symbolic_ref("-q", "--short", "HEAD").strip()
    except GitCommandError:
        return repo.git.rev_parse("--short", "HEAD").strip()


def _advance_sync_tag(repo: Any, tag_name: str, commit: str) -> None:
    """Move the sync tag to the commit just cherry-picked from the source history."""
    repo.git.tag("-f", tag_name, commit)
    print(f"moved tag {tag_name!r} to {commit[:12]}")


def _as_text(data: str | bytes | None) -> str:
    if data is None:
        return ""
    if isinstance(data, bytes):
        return data.decode("utf-8", errors="replace")
    return data


def _write_failed_pick_head(repo: Any, commit_hex: str) -> None:
    root = repo.working_tree_dir
    if root is None:
        return
    path = Path(root) / ESP_REBASE_HEAD_FILE
    path.write_text(commit_hex.strip() + "\n", encoding="ascii")


def _esp_rebase_head_path(repo: Any) -> Path | None:
    root = repo.working_tree_dir
    if root is None:
        return None
    return Path(root) / ESP_REBASE_HEAD_FILE


def _read_esp_rebase_head_hash(repo: Any) -> str:
    path = _esp_rebase_head_path(repo)
    if path is None or not path.is_file():
        _die(
            f"{ESP_REBASE_HEAD_FILE} not found at repository root; nothing to advance the tag to"
        )
    raw = path.read_text(encoding="utf-8", errors="replace").strip()
    line = raw.splitlines()[0] if raw else ""
    line = line.strip()
    if not line or not re.fullmatch(r"[0-9a-f]{40}", line, re.I):
        _die(
            f"{ESP_REBASE_HEAD_FILE} does not contain a valid full commit hash (got {line!r})"
        )
    try:
        return repo.git.rev_parse("--verify", f"{line}^{{commit}}").strip()
    except GitCommandError:
        _die(f"hash in {ESP_REBASE_HEAD_FILE} is not a valid commit: {line!r}")


def _git_cherry_pick_continue_no_editor(repo: Any) -> None:
    """Run `git cherry-pick --continue` with a no-op editor (non-interactive)."""
    keys = ("GIT_EDITOR", "EDITOR", "VISUAL", "SEQUENCE_EDITOR")
    saved = {k: os.environ.get(k) for k in keys}
    try:
        for k in keys:
            os.environ[k] = "true"
        repo.git.cherry_pick("--continue")
    finally:
        for k in keys:
            v = saved.get(k)
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def _wip_br_finish(
    repo: Any,
    failed_full_hash: str,
    dest_base_short: str,
    sync_tag_target_short: str,
) -> str:
    """Complete WIP pick, create sync_fail branch at HEAD, then drop that commit on current branch."""
    if repo.working_tree_dir is None:
        _die("bare repository: --wip-br is not supported")
    repo.git.add("-u")
    try:
        _git_cherry_pick_continue_no_editor(repo)
    except GitCommandError as e:
        _print_git_command_error(e)
        _die("git cherry-pick --continue failed after staging; fix manually")
    try:
        commit_short = repo.git.rev_parse("--short=7", failed_full_hash).strip()
    except GitCommandError:
        commit_short = failed_full_hash[:7]
    if not re.fullmatch(r"[0-9a-f]+", commit_short, re.I):
        commit_short = failed_full_hash[:7]
    base = dest_base_short.strip()
    tag_s = sync_tag_target_short.strip()
    for label, h in (("dest base", base), ("sync tag", tag_s), ("commit", commit_short)):
        if not re.fullmatch(r"[0-9a-f]+", h, re.I):
            _die(f"invalid {label} short hash for WIP branch name: {h!r}")
    #branch = f"sync_fail/{base}_{tag_s}_{commit_short}"
    branch = f"sync_fail/{tag_s}_{commit_short}"
    try:
        repo.git.branch("-f", branch, "HEAD")
    except GitCommandError as e:
        _die(f"git branch -f {branch!r} HEAD failed: {e}")
    root = Path(repo.working_tree_dir)
    (root / ESP_REBASE_WIP_BRANCH_FILE).write_text(branch + "\n", encoding="utf-8")
    try:
        repo.git.reset("--hard", "HEAD~1")
    except GitCommandError as e:
        _die(
            f"WIP branch {branch!r} was created, but git reset --hard HEAD~1 on the current "
            f"branch failed: {e}. Remove the conflict commit manually if needed; WIP still points "
            f"to the recorded commit."
        )
    return branch


def _print_git_command_error(e: BaseException) -> None:
    """Print stdout/stderr from a failed git invocation (GitPython GitCommandError)."""
    stdout = _as_text(getattr(e, "stdout", None)).strip()
    stderr = _as_text(getattr(e, "stderr", None)).strip()
    if stdout:
        print("--- git cherry-pick (stdout) ---", file=sys.stderr)
        print(stdout, file=sys.stderr)
    if stderr:
        print("--- git cherry-pick (stderr) ---", file=sys.stderr)
        print(stderr, file=sys.stderr)
    if not stdout and not stderr:
        msg = str(e).strip()
        if msg:
            print("--- git cherry-pick ---", file=sys.stderr)
            print(msg, file=sys.stderr)


def _handle_pick_failure(
    repo: Any,
    tag: str,
    *,
    no_abort: bool,
    picked_count: int,
    failed_pick_commit: str | None,
    context: str,
) -> NoReturn:
    if failed_pick_commit:
        _write_failed_pick_head(repo, failed_pick_commit)

    if _cherry_pick_in_progress(repo):
        if no_abort:
            print(
                f"Failed '{context}' was not aborted. Resolve conflicts and re-run script with -c/--continue.",
                file=sys.stderr,
            )
        else:
            try:
                repo.git.cherry_pick("--abort")
            except GitCommandError:
                print(
                    "warning: git cherry-pick --abort failed",
                    file=sys.stderr,
                )
    else:
        print(
            f"Failed to execute '{context}'. Run 'git cherry-pick {failed_pick_commit}' and check for errors. "
            "When errors are fixed re-run this script.",
            file=sys.stderr,
        )
    raise SystemExit(1 if picked_count == 0 else 2)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Cherry-pick range from tag to source HEAD onto destination."
    )
    parser.add_argument("source_branch", help="Branch to take commits from")
    parser.add_argument("dest_branch", help="Branch to apply commits onto")
    parser.add_argument(
        "tag",
        nargs="?",
        default=DEFAULT_TAG,
        help=f"Exclusive lower bound (default: {DEFAULT_TAG})",
    )
    parser.add_argument(
        "-c",
        "--continue",
        dest="do_continue",
        action="store_true",
        help=(
            "Run git cherry-pick --continue first, then advance sync tag from "
            f"{ESP_REBASE_HEAD_FILE}, then cherry-pick remaining commits."
        ),
    )
    parser.add_argument(
        "--no-abort",
        action="store_true",
        default=False,
        help=(
            "On failure, do not run git cherry-pick --abort; leave the "
            "repository in the failed cherry-pick state for manual fix."
        ),
    )
    parser.add_argument(
        "-b",
        "--wip-br",
        action="store_true",
        default=False,
        help=(
            "Only with --no-abort. On conflict: stage, cherry-pick --continue, create WIP branch "
            "sync_fail/<dest_BASE>_<tag_TARGET>_<failed_COMMIT> at that commit, then "
            "git reset --hard HEAD~1 on the current branch, and stop without advancing the sync tag."
        ),
    )
    args = parser.parse_args()

    if args.wip_br and not args.no_abort:
        _die("--wip-br requires --no-abort")

    if git is None:
        print(
            "error: install GitPython: pip install GitPython",
            file=sys.stderr,
        )
        raise SystemExit(1)

    source_branch = args.source_branch
    dest_branch = args.dest_branch
    tag = args.tag

    try:
        repo = git.Repo(search_parent_directories=True)
    except InvalidGitRepositoryError:
        _die("not inside a git repository")

    source_ref = _rev_parse_verify(repo, source_branch)
    dest_original_head = _rev_parse_verify(repo, dest_branch)
    try:
        dest_base_short = repo.git.rev_parse("--short=7", dest_original_head).strip()
    except GitCommandError:
        dest_base_short = dest_original_head[:7]
    if not re.fullmatch(r"[0-9a-f]+", dest_base_short, re.I):
        dest_base_short = dest_original_head[:7]

    tag_commit = _tag_commit_sha(repo, tag)
    try:
        sync_tag_target_short = repo.git.rev_parse("--short=7", tag_commit).strip()
    except GitCommandError:
        sync_tag_target_short = tag_commit[:7]
    if not re.fullmatch(r"[0-9a-f]+", sync_tag_target_short, re.I):
        sync_tag_target_short = tag_commit[:7]
    try:
        repo.git.merge_base("--is-ancestor", tag_commit, source_ref)
    except GitCommandError:
        _die(
            f"tag {tag!r} ({tag_commit[:12]}) is not contained in branch {source_branch!r}"
        )

    orig_branch: str | None = None
    if not repo.head.is_detached:
        orig_branch = repo.active_branch.name

    if args.do_continue:
        try:
            repo.git.cherry_pick("--continue")
        except GitCommandError as e:
            print(
                "error: git cherry-pick --continue failed",
                file=sys.stderr,
            )
            _print_git_command_error(e)
            failed = _failed_pick_commit_ref(repo) or ""
            _handle_pick_failure(
                repo,
                tag,
                no_abort=args.no_abort,
                picked_count=0,
                failed_pick_commit=failed or None,
                context="cherry-pick --continue",
            )

        head_hash = _read_esp_rebase_head_hash(repo)

        _advance_sync_tag(repo, tag, head_hash)
        # path = _esp_rebase_head_path(repo)
        # if path is not None and path.is_file():
        #     try:
        #         path.unlink()
        #     except OSError as e:
        #         print(
        #             f"warning: could not remove {ESP_REBASE_HEAD_FILE}: {e}",
        #             file=sys.stderr,
        #         )

    repo.git.checkout(dest_branch)

    picked_count = 0
    tag_commit = _tag_commit_sha(repo, tag)
    commits = _commits_between(repo, tag_commit, source_ref)
    if not commits:
        print(f"nothing to cherry-pick: no commits between {tag!r} and {source_branch!r}")
        raise SystemExit(0)

    pick_failed = False
    failed_pick_commit: str | None = None
    for commit in commits:
        subj = repo.git.log("-1", "--format=%s", commit)
        print(f"cherry-picking {commit[:12]} {subj}")
        try:
            repo.git.cherry_pick(commit)
        except GitCommandError as e:
            print(
                f"error: cherry-pick conflicted or failed for {commit}",
                file=sys.stderr,
            )
            _print_git_command_error(e)
            if (
                args.wip_br
                and _cherry_pick_in_progress(repo)
            ):
                _write_failed_pick_head(repo, commit)
                wip = _wip_br_finish(
                    repo, commit, dest_base_short, sync_tag_target_short
                )
                print(
                    f"Created WIP branch {wip!r} (it keeps the conflict cherry-pick). "
                    f"Removed that commit from the current branch (git reset --hard HEAD~1). "
                    f"Sync tag {tag!r} was not advanced; remaining commits were not cherry-picked.",
                )
                raise SystemExit(1 if picked_count == 0 else 2)
            pick_failed = True
            failed_pick_commit = commit
            break
        picked_count += 1
        _advance_sync_tag(repo, tag, commit)

    if pick_failed:
        _handle_pick_failure(
            repo,
            tag,
            no_abort=args.no_abort,
            picked_count=picked_count,
            failed_pick_commit=failed_pick_commit,
            context="cherry-pick",
        )

    cur = _current_branch_label(repo)
    if orig_branch:
        print(f"done. Current branch: {cur}. Original branch was: {orig_branch}")
    else:
        print(f"done. Current branch: {cur}")


if __name__ == "__main__":
    main()
