# Development Guidelines

This document contains critical information about working with this codebase. Follow these guidelines precisely.

## Environment Setup

To ensure everyone uses the same environment, follow these steps:

1. **Initial Setup**: Run `uv sync` to create/update your environment from the lockfile
2. **After Pulling Changes**: If `uv.lock` has changed, run `uv sync` again
3. **Adding Dependencies**: Use `uv add <package>` which updates both pyproject.toml and uv.lock
4. **Removing Dependencies**: Use `uv remove <package>`

The `uv.lock` file ensures all developers and CI/CD systems use exactly the same package versions.

## Project State Management (PROJECT.md)

You are required to maintain a `PROJECT.md` file in the root directory to track the repository's evolution.

1. **Update Timing**: Update this file **ONLY** after a task, feature, or bug fix is fully completed and verified.
   - **STRICT PROHIBITION**: Never update this file before work is done. It is a record of history, not a plan.
2. **Required Content**:
   - **Project/Repo Outline**: A high-level map of the current architecture and modules.
   - **Progress**: A summary of what has been achieved in the latest iteration.
   - **Lessons Learned**: Critical discoveries, patterns, or "gotchas" encountered during implementation.
   - **Fixed Bugs**: A log of resolved issues (include specific error messages or behavior fixed).

## Core Development Rules

1. **Package Management**
   - ONLY use uv, NEVER pip
   - Environment setup: `uv sync`
   - Installation: `uv add package`
   - Upgrading: `uv add --dev package --upgrade-package package`

2. **Code Style**
    - PEP 8 naming (snake_case for functions/variables)
    - Class names in PascalCase
    - Constants in UPPER_SNAKE_CASE
    - Document with docstrings
    - Use f-strings for formatting

3. **Commit Standards**
    - For user-reported bugs/features: `git commit --trailer "Reported-by:<name>"`
    - For Github issues: `git commit --trailer "Github-Issue:#<number>"`
    - **FORBIDDEN**: Never mention `co-authored-by` or the AI tool used in commit messages/PRs.

## Development Philosophy

- **Simplicity**: Write simple, straightforward code
- **Reliability**: Code must be robust against anomalies (heavy use of assertions)
- **Readability**: Make code easy to understand; logic must be linear (no recursion)
- **Maintainability**: Write code that's easy to update
- **Less Code = Less Debt**: Minimize code footprint

## Coding Best Practices

- **Early Returns**: Use to avoid nested conditions
- **Descriptive Names**: Use clear variable/function names (prefix handlers with "handle")
- **Constants Over Functions**: Use constants where possible
- **DRY Code**: Don't repeat yourself
- **Minimal Changes**: Only modify code related to the task at hand
- **Function Ordering**: Define composing functions before their components
- **Build Iteratively**: Start with minimal functionality and verify it works before adding complexity
- **Clean Logic**: Keep core logic clean and push implementation details to the edges
- **File Organization**: Balance file organization with simplicity
- **TODO Comments**: Mark issues in existing code with "TODO:" prefix
- **Comments**: Only for stuff that is ambiguous, NOT marks for changes

## Pull Requests

- Create a detailed message of what changed. Focus on the high level description of the problem it tries to solve, and how it is solved.
- **FORBIDDEN**: Never mention `co-authored-by` or the AI tool used in commit messages/PRs.

## Python Tools & Formatting

1. **Ruff**
   - Format: `uv run ruff format .`
   - Check: `uv run ruff check . --fix`
   - Configuration: Line length 150 chars

2. **Pre-commit Hooks**
   - **Standard**: checks for yaml, toml, json, large files, and shebangs.
   - **Ruff**: Runs both check (fix) and format (line-length 99).
   - **PyUpgrade**: Automatically upgrades syntax for newer Python versions.
   - **Gitleaks**: Scans for accidental secret leaks.

3. **Type Checking**
   - Tool: `uv run ty`
   - Install: `uv add --dev ty`
   - Requirements: Explicit None checks for Optional, type narrowing.

4. **Error Resolution**
   - Check git status before commits
   - Run formatters before type checks
   - Keep changes minimal
   - Document public APIs
