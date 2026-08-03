---
name: pre-pr-checks
description: >
  Run the checks CI enforces before opening or updating a pull request on
  torch_fl. Use this whenever you are about to `gh pr create`, push to a PR
  branch, or amend a commit that is already on a PR — and when a PR's `lint /
  Lint` job has gone red. Covers: the exact pinned ruff version CI uses, the
  two ruff commands that gate every job downstream, which interpreter to run
  them with on vendor boxes, and how to update an existing PR afterwards.
---

# Pre-PR checks (torch_fl)

## Why this exists

`.github/workflows/ci.yml` runs `lint` **first** and every other job declares
`needs: lint`. A formatting slip therefore does not just fail one check — it
skips the build and integration jobs entirely, so a red lint tells you nothing
about whether the actual change works. Lint is cheap (seconds) and it gates
everything, so run it before you push, not after CI complains.

## The gate

`.github/workflows/lint.yml` is the whole contract. Two commands, one pinned
tool version:

```bash
pip install ruff==0.15.12       # pin matters: formatting rules shift between
                                # ruff releases, so an unpinned local ruff can
                                # disagree with CI in both directions
ruff check .                    # lint rules
ruff format --check .           # formatting
```

Both must pass. Run them from the repo root — the config lives in
`pyproject.toml` and the paths are relative.

To fix rather than just report:

```bash
ruff check --fix .
ruff format .
```

Then re-run the two `--check` forms to confirm, and re-run any tests covering
files that were reformatted. Formatting should be semantically inert, but
confirming costs one command.

## Interpreter caveat on vendor boxes

On machines where the vendor torch lives in the **system** interpreter rather
than whatever `python` resolves to (Hygon DCU/DTK is one — bare `python` there
is a conda install with no torch), spell the interpreter out or you will
install ruff into the wrong environment and get "No module named ruff":

```bash
/usr/bin/python3 -m pip install ruff==0.15.12
/usr/bin/python3 -m ruff check .
/usr/bin/python3 -m ruff format --check .
```

## Reading a red lint job

`ruff format --check` reports only a count ("1 file would be reformatted") —
it does not name the file in the CI summary. Get the specifics locally:

```bash
ruff format --check .                        # names the files
ruff format --diff path/to/file.py           # shows the exact required change
```

Or pull the CI log directly:

```bash
gh run view <run-id> --repo <owner>/<repo> --log-failed
gh pr checks <pr-number> --repo <owner>/<repo>
```

## Updating a PR after fixing

This repo's PRs land as a **single commit**. Fold lint fixes into the existing
commit rather than stacking a "fix lint" commit on top:

```bash
git add <files>
git commit --amend --no-edit
git push --force-with-lease origin <branch>
```

`--force-with-lease` (not `--force`) so the push aborts if someone else has
pushed to the branch meanwhile. Force-pushing is appropriate here because the
branch is your own PR branch; it is not appropriate for shared or upstream
branches.

## Checklist

Before `gh pr create` or any push to a PR branch:

1. `ruff check .` → "All checks passed!"
2. `ruff format --check .` → "N files already formatted", nothing to reformat
3. Tests covering your change still pass (and re-run any file ruff reformatted)
4. `git diff --stat` against the base — confirm the diff contains only what you
   intended, especially when generated files are involved
