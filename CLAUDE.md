# Project conventions for Claude Code

## Language

**All GitHub-facing text must be written in English** — PR titles, PR descriptions,
commit messages, issue text, and code review comments. This repository's code,
comments, and existing history are all English, and PRs are read by contributors
who do not read Chinese.

**Everything under `docs/` must be written in English**, without exception —
design docs, plans, vendor integration notes, analysis write-ups. A doc drafted in
Chinese during a working session must be translated before it lands. The one
allowance is an explicitly localized top-level README (`README_zh.md`), which
exists as a translation of the English `README.md`.

Chat replies to the user in this session stay in whatever language the user is
using (usually Chinese). The rule is about what gets committed and published, not
about how we talk here.

If a PR description has already been opened in the wrong language, fix it with
`gh api -X PATCH repos/{owner}/{repo}/pulls/{n} -f body=@file` rather than
leaving it and noting the problem.

## Simplified workflow for routine tasks

**For routine development tasks, skip the full superpowers design workflow and implement
directly.** Only invoke brainstorming/writing-plans for tasks that genuinely require
upfront design:

- Complex feature additions with multiple valid architectural approaches
- Performance optimization requiring measurement and trade-off analysis
- Large-scale refactoring affecting many files or core abstractions
- Tasks where the user explicitly requests a design discussion

**Implement directly for:**
- Bug fixes with clear root cause
- Straightforward feature additions with obvious implementation
- Code cleanup and file reorganization
- Documentation updates
- Test additions
- Dependency updates

When implementing directly, still follow investigation-before-action (read relevant code
first) and verification-before-completion (run tests, check the build). The difference is
skipping the separate design phase when the path forward is clear.
