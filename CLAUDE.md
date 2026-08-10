# Project conventions for Claude Code

## Language

**All GitHub-facing text must be written in English** — PR titles, PR descriptions,
commit messages, issue text, and code review comments. This repository's code,
comments, and existing history are all English, and PRs are read by contributors
who do not read Chinese.

Chat replies to the user in this session stay in whatever language the user is
using (usually Chinese). The rule is about what gets published to GitHub, not
about how we talk here.

If a PR description has already been opened in the wrong language, fix it with
`gh api -X PATCH repos/{owner}/{repo}/pulls/{n} -f body=@file` rather than
leaving it and noting the problem.
