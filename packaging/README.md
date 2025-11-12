Packaging Automation
====================

Use the helper script in this directory to build wheels (via ``maturin``) and kick off a Conda recipe build. The script orchestrates the commands described in :doc:`../docs/release-strategy`.

## Requirements

- Python 3.9+
- ``maturin`` and ``cargo`` on ``PATH``
- Optional: ``conda-build`` (for the Conda recipe)

## Building Wheels

```bash
python packaging/build.py wheel --release
```

Outputs land in ``dist/wheels``. Pass ``--target <triple>`` to cross-compile if you have the appropriate toolchain.

## Building the Conda Package

```bash
python packaging/build.py conda --channel diffsol --output-folder dist/conda
```

This wraps ``conda-build packaging/conda``. The recipe installs the freshly built wheel so you can publish a Conda package that mirrors the PyPI artifacts.

## Combined Builds

```bash
python packaging/build.py all --release --channel diffsol
```

Runs wheel + Conda builds in sequence, sharing the same output directories.

See ``packaging/build.py --help`` for full CLI usage.
