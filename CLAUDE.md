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

## GitHub Templates and Issue/PR Guidelines

**When creating issues or pull requests, use the appropriate templates:**

### For AI Agents (Claude Code, Codex, etc.)

**READ THESE FIRST before filing any issue or PR:**
1. `.github/AI_AGENT_GUIDE.md` - Complete guidelines for AI agents
2. `.github/CLAUDE_CODE_GUIDE.md` - Claude Code specific instructions

**Required templates:**
- **Issues**: Use `.github/ISSUE_TEMPLATE/ai_agent_issue.md`
- **Pull Requests**: Use `.github/PULL_REQUEST_TEMPLATE/ai_agent_pr.md`

**Pre-submission validation:**
```bash
# Run this before creating any PR
python scripts/validate_ai_pr.py --pr-body pr_description.md
```

**Mandatory requirements for AI PRs:**
- Root cause analysis (not just symptoms)
- Investigation process documented
- Actual test output pasted (not claims like "tests pass")
- Linting results shown (`ruff check`, `ruff format --check`)
- Edge cases identified and tested
- Human reviewer assigned
- Everything in English (enforced by validation script)

**Creating an AI PR:**
```bash
# After implementation and validation
gh pr create \
  --title "fix: <description>" \
  --body-file pr_description.md \
  --label "ai-generated"
```

### For Human Contributors

Use standard templates:
- Bug reports: `.github/ISSUE_TEMPLATE/bug_report.md`
- Feature requests: `.github/ISSUE_TEMPLATE/feature_request.md`
- Platform support: `.github/ISSUE_TEMPLATE/platform_support.md`
- Operator requests: `.github/ISSUE_TEMPLATE/operator_support.md`
- Pull requests: `.github/PULL_REQUEST_TEMPLATE.md`

See `CONTRIBUTING.md` for detailed contribution guidelines.

## Commit Message Format

All commits should follow this format:
```
<type>: <subject>

<body>

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>
```

**Valid types:**
- `feat:` - New feature
- `fix:` - Bug fix
- `perf:` - Performance improvement
- `refactor:` - Code refactoring (no behavior change)
- `docs:` - Documentation changes
- `test:` - Test additions or modifications
- `ci:` - CI/CD changes
- `build:` - Build system changes

**Example:**
```
fix: resolve CUDA stream synchronization race in profiler

The CUPTI callbacks must be registered before the first CUDA context
is created. This change moves callback registration to module import
time, ensuring it happens before any CUDA operations.

Tested: pytest tests/integration/test_profiler.py - all pass

Fixes #123

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>
```

## Code Quality Requirements

### Pre-commit Checks

Before committing any code changes:
```bash
# Linting (required)
ruff check
ruff format --check

# If format issues found, fix them:
ruff format

# Run relevant tests
pytest tests/unit/ -v
pytest tests/integration/ops/test_<relevant>.py -v
```

### Code Style

- Follow existing code style in the files you modify
- Match comment density of surrounding code
- Reuse existing utilities instead of reinventing
- Keep changes focused (avoid unnecessary refactoring)

## Testing Requirements

### Test Organization

- Unit tests: `tests/unit/` - Fast tests (< 1s each)
- Integration tests: `tests/integration/` - Full operator/model tests
- Use pytest marks to indicate platform requirements:
  - `@pytest.mark.anyplatform` - Runs on all platforms
  - `@pytest.mark.cuda` - CUDA specific
  - `@pytest.mark.ascend` - Ascend NPU specific
  - `@pytest.mark.flaggems` - FlagGems backend required

### Verification Before Completion

After implementing changes:
1. Run linting (ruff check, ruff format --check)
2. Run relevant unit tests
3. Run relevant integration tests
4. Manually verify the fix/feature works
5. Check that no debug code remains (no print statements, TODOs)

## Platform Considerations

When making changes, consider impact on all platforms:
- **CUDA** - Primary development platform
- **MetaX** - Boxing mode (reuses CUDA kernels)
- **Ascend** - ACL-based backend
- **PPU** - CUDA-compatible via PPU SDK

If a change only affects one platform, document why in the PR.

## Documentation

Update documentation when:
- Adding new features → Update README
- Changing APIs → Update docstrings
- Platform changes → Update platform-specific sections
- New conventions → Update this file (CLAUDE.md)

All documentation must be in English.

### Operator support records

Any change that adds, enables, removes, disables, or reroutes an operator through
a vendor-native backend or FlagGems must update
`docs/reference/operator-support.md` for every affected hardware platform in the
same change. Rerun `tests/manual/flaggems_overload_survey.py` on the affected
hardware and update the summary, raw evidence, provenance, and update history
from measured results. Do not infer support from routing configuration alone.

If affected hardware is unavailable, mark its data as **not revalidated** and
record the evidence gap in both the support report and the PR. Do not silently
retain old results as though they were measured against the new cohort.

## Related Documentation

- **CONTRIBUTING.md** - Detailed contribution workflow
- **README.md** - Project overview and build instructions
- **.github/AI_AGENT_GUIDE.md** - AI agent specific guidelines
- **.github/CLAUDE_CODE_GUIDE.md** - Claude Code integration guide
- **.github/TEMPLATES.md** - Template selection guide

