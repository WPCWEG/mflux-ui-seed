# justfile for the mflux-ui-seed Python 3.12+ project, matching Yanlume Python 3.12.15
# Adapted from the mflux core justfile: github.com/mflux-community/mflux (the plumbing this UI seed uses).
# The mflux.web namespace model this seed targets is defined by mflux PR #776.

set shell := ["bash", "-euo", "pipefail", "-c"]

python_version := "3.12.15"
venv_dir := ".venv"
# Ruff version derives from the pinned dev dependency in pyproject.toml (single source of truth)
ruff_version := `sed -n 's/^    "ruff==\([0-9.]*\)",$/\1/p' pyproject.toml`

# Show all recipes
default:
    @just --list

# Create the virtual environment, install dependencies and pre-commit hooks
venv-install: venv-init ensure-pre-commit
    @echo "🏗️ Checking current directory is a git repo..."
    @if git rev-parse --is-inside-work-tree >/dev/null 2>&1; then \
      echo "✅ Directory is a Git repository"; \
    else \
      git init; \
      echo "✅ ran git init"; \
    fi
    @echo "🏗️ Installing dependencies and pre-commit hooks..."
    uv sync --locked --python {{ python_version }}
    @echo "✅ Dependencies installed."
    pre-commit install
    @echo "✅ Pre-commit hooks installed."

# Create the Python virtual environment with uv
venv-init: expect-arm64 expect-uv
    @echo "🏗️ Creating virtual environment with recommended uv tool:"
    uv python install --quiet {{ python_version }}
    uv venv --clear --python {{ python_version }}
    @echo "✅ Python {{ python_version }} virtual environment created at {{ venv_dir }}"

# Run ruff linter (read-only; use 'just dev-check' to auto-apply fixes)
dev-lint:
    @echo "🏗️ Running linters (ruff {{ ruff_version }}, pinned via pyproject.toml), your files will not be mutated."
    uvx ruff@{{ ruff_version }} check
    @echo "✅ Linting complete."

# Lint the justfile itself (fails if 'just --fmt' would reformat it; run 'just --fmt' to fix)
dev-lint-justfile:
    {{ just_executable() }} --fmt --check

# Format the justfile in place
# dev-format-justfile:
# {{ just_executable() }} --fmt

# Run ruff code formatter (mutates files; review your git diffs after)
dev-format:
    @echo "🏗️ Running formatter (ruff {{ ruff_version }}, pinned via pyproject.toml), your files will be changed to comply to formatting configs."
    uvx ruff@{{ ruff_version }} format
    git diff --stat
    @echo "✅ Formatting complete. Please review your git diffs, if any."

# Run ty type checker (version pinned via pyproject.toml dev deps + uv.lock)
dev-typecheck:
    @echo "🏗️ Running ty type checker (pinned via pyproject.toml)..."
    uv sync --locked --all-extras
    uv run --no-sync ty check
    @echo "✅ Type checking complete."

# Run pre-commit auto-fixes and formatters on all files
dev-check:
    @echo "🏗️ Running pre-commit linter and formatters on files..."
    pre-commit run --all-files

# Run the test suite
test: _test-run

# Start the web server (pass mflux-web options, e.g. 'just run --port 8002')
run *args:
    uv run mflux-web {{ args }}

# Build the wheel and sdist into dist/ (deletes old files in dist/ first)
build:
    uv build --clear

# Remove the virtual environment
venv-uninstall:
    @echo "🧼 Cleaning up venv."
    rm -rf {{ venv_dir }}
    @echo "✅ Cleaned up venv. Run 'just venv-install' to re-generate."

# --- private helpers ----------------------------------------------------------

# 🖥️ mflux and MLX are known to be compatible with arm64/aarch64 Mac and Linux only
# (host arch via uname, not just's build arch — an x86_64 just under Rosetta must not refuse)
[private]
expect-arm64:
    @case "$(uname -m)" in \
        arm64 | aarch64) ;; \
        *) echo "mflux and MLX (the plumbing this UI seed builds on) are known to be compatible with arm64/aarch64 Mac and Linux only (detected: $(uname -m)). This justfile does not support your machine."; exit 1 ;; \
    esac

# we "expect" uv but should not install it for the user, let user *choose* to trust a third party installer
[private]
expect-uv:
    @if ! command -v uv > /dev/null; then \
        echo "You can use classic python -m venv to setup this project,"; \
        echo "but we officially support using uv for managing this project's environment."; \
        echo ""; \
        echo "Please install uv to continue:"; \
        echo "    https://github.com/astral-sh/uv?tab=readme-ov-file#installation"; \
        exit 1; \
    fi

# assume reasonably pre-commit is a safe dependency given its wide support (e.g. GitHub Actions integration)
[private]
ensure-pre-commit:
    @if ! command -v pre-commit > /dev/null; then \
        echo "pre-commit required for submitting commits before pull requests. Using uv tool to install pre-commit."; \
        uv tool install pre-commit; \
    fi

# shared test runner: locked env incl. dev extras (pinned mflux) for testing
[private]
_test-run args="":
    @echo "🏗️ Syncing locked environment with dev extras..."
    uv sync --locked --all-extras
    @echo "🏗️ Running pytest (see command line for selector)..."
    uv run --no-sync python -m pytest {{ args }}
    @echo "✅ Tests completed"
