Release & Versioning Strategy
=============================

diffsol follows semantic versioning (``MAJOR.MINOR.PATCH``) across both the Rust core library and the PyTorch bindings. This document records compatibility guarantees, planned migration paths, and policies for deprecating APIs.

Semantic Versioning
-------------------

- **MAJOR** – breaking API changes (DiffSL syntax, Python/Rust function signatures, solver defaults). Requires an RFC and two release candidates.
- **MINOR** – backward-compatible features (new solvers, additional arguments, docs, benchmarks).
- **PATCH** – bug fixes, performance improvements, documentation corrections.

Compatibility Matrix
--------------------

.. list-table::
   :header-rows: 1

   * - diffsol-pytorch
     - diffsol (core)
     - PyTorch
     - Python
     - Notes
   * - 0.1.x
     - 0.7.x
     - 1.11 – 2.2
     - 3.9 – 3.13
     - Built with PyO3 0.25, Cranelift backend by default
   * - 0.2.x (planned)
     - 0.8.x
     - 2.0 – 2.3
     - 3.10 – 3.13
     - Adds prebuilt wheels, LLVM/Enzyme optional feature flag

Migration Planning
------------------

- **DiffSL syntax** – we maintain backward compatibility for existing ``u_i``/``F_i`` blocks. When introducing new constructs we keep the old behaviour behind feature flags until the next MAJOR release.
- **Python API** – new keyword arguments are added with defaults; removing arguments or changing return shapes requires a deprecation cycle (see below).
- **Solver behaviour** – changes to default tolerances/solvers are announced at least one MINOR release ahead, with environment variables to opt into the new behaviour early.

Deprecation Policy
------------------

1. Mark the API with a ``DeprecationWarning`` in Python or ``#[deprecated]`` in Rust.
2. Document the replacement in ``docs/release-strategy.rst`` and ``CHANGELOG.md``.
3. Maintain shims for at least one MINOR release (e.g., 0.2.x introduces the new API, 0.3.x removes the old one).

Release Checklist
-----------------

1. Ensure ``maturin develop`` and ``pytest`` pass on Linux/macOS/Windows.
2. Update ``benchmarks/results.md`` with fresh numbers.
3. Bump versions in ``Cargo.toml`` and ``pyproject.toml``.
4. Tag ``vMAJOR.MINOR.PATCH`` and publish crates/wheels.
5. Announce in GitHub Discussions + Matrix.
6. Archive the release notes under ``docs/release-strategy.rst`` for future reference.

Future Work
-----------

- Publish prebuilt wheels (Linux/macOS) once LLVM detection is automated.
- Introduce per-solver semantic versions (e.g., ``diffsol-bdf`` crate) to decouple GPU work.
- Provide long-term-support (LTS) releases once the API stabilises post-1.0.
