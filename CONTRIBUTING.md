# Contributing to diffsol-pytorch

We welcome pull requests, issues, and community discussions! This document explains how to set up a development environment, follow the project’s style guidelines, and propose changes responsibly.

## Getting Started

1. **Clone & bootstrap**

   ```bash
   git clone https://github.com/martinjrobins/diffsol.git
   cd diffsol
   python -m venv .venv
   .\.venv\Scripts\activate          # Linux/macOS: source .venv/bin/activate
   pip install -r docs/requirements.txt maturin pytest black ruff
   ```

2. **Build the PyTorch bindings**

   ```bash
   cd bindings/diffsol-pytorch
   maturin develop
   cd -
   ```

3. **Run the test suite**

   ```bash
   .\.venv\Scripts\python.exe -m pytest bindings/diffsol-pytorch/tests
   ```

   Tests that require LLVM/Enzyme automatically skip when the build only contains the Cranelift backend.

## Coding Standards

- **Rust**: use ``cargo fmt`` and ``cargo clippy --all-targets --all-features``. The main crate lives under ``diffsol/``; PyO3 bindings live under ``bindings/diffsol-pytorch/src``.
- **Python**: format with ``black`` (line length 100) and lint with ``ruff``. Run both via:

  ```bash
  black bindings/diffsol-pytorch examples docs
  ruff check bindings/diffsol-pytorch examples docs
  ```

- **DiffSL**: keep equations in the ``u_i`` / ``F_i`` form for clarity; include comments explaining physical meaning when adding new files to ``examples/``.
- **Error handling**: prefer descriptive ``RuntimeError`` or ``PyValueError`` messages that tell the user how to fix their input. In Rust, use ``thiserror`` and map errors into PyO3’s typed exceptions.

## Pull Requests

1. Create an issue (or pick one up) before opening a PR.
2. Fork + branch (e.g., ``feature/sparse-jacobians``).
3. Make changes with tests/docs.
4. Run the full test suite plus ``maturin develop`` to ensure the extension still builds.
5. Update ``README.md`` or ``docs/`` if behaviour changes.
6. Open a PR and fill in the template (tests run, docs updated, breaking changes called out).

PRs that alter APIs must describe the migration path and, when possible, include deprecation warnings.

## Reporting Bugs & Requesting Features

Use the GitHub issue templates under ``.github/ISSUE_TEMPLATE``. For bugs please attach:

- diffsol and PyTorch versions (``python -c "import torch, diffsol_pytorch; print(torch.__version__)"``)
- Platform (OS, compiler)
- Minimal DiffSL snippet reproducing the problem
- Exact command / stack trace

## Release & Versioning

We follow semantic versioning:

- ``MAJOR`` bumps for breaking API changes
- ``MINOR`` for additive features
- ``PATCH`` for bug fixes

Keep the ``release-strategy`` document updated with compatibility tables and migration notes whenever your change affects public APIs.

## Community

- Discussions: https://github.com/martinjrobins/diffsol/discussions
- Real-time chat (Matrix/Discord bridge): see ``docs/community.rst`` for invite links.
- Meetings: monthly community call (announced on Discussions).

When in doubt, ask in Discussions or tag a maintainer in your issue. We’re excited to build diffsol together!
