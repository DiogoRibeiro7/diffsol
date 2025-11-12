- ``Dockerfile`` – CPU-only base for rapid iteration (Cranelift build).
- ``Dockerfile.gpu`` – CUDA-enabled build that layers on top of ``nvidia/cuda``.
- ``Dockerfile.llvm`` – Ubuntu 22.04 image with LLVM/Enzyme toolchain for feature-complete builds (reverse/forward sensitivities).

## Building

```bash
docker build -f docker/Dockerfile -t diffsol-pytorch:cpu .
docker build -f docker/Dockerfile.gpu -t diffsol-pytorch:cuda .
```

Mount your workspace when running to iterate on examples or docs:

```bash
docker run --rm -it -v ${PWD}:/workspace diffsol-pytorch:cpu python examples/integration/neural_ode/train.py
```

## LLVM/Enzyme image

The new ``Dockerfile.llvm`` installs LLVM 18, Enzyme prerequisites, Rust, and Python tooling. Use it to build wheels with ``--features llvm,python`` and run the full test suite:

```bash
# Build the toolchain image once
docker build -f docker/Dockerfile.llvm -t diffsol-pytorch:llvm .

# Install the bindings with LLVM+Enzyme and run tests
docker run --rm -v ${PWD}:/workspace -w /workspace diffsol-pytorch:llvm \
  bash -lc "maturin develop --locked --manifest-path bindings/diffsol-pytorch/Cargo.toml --release --no-default-features --features llvm,python && pytest bindings/diffsol-pytorch/tests"

# Run an end-to-end example (neural ODE) inside the same environment
docker run --rm -v ${PWD}:/workspace -w /workspace diffsol-pytorch:llvm \
  bash -lc \"maturin develop --locked --manifest-path bindings/diffsol-pytorch/Cargo.toml --release --no-default-features --features llvm,python && python examples/integration/neural_ode/train.py --samples 32 --epochs 1\"

# Produce manylinux-style wheels
docker run --rm -v ${PWD}:/workspace -w /workspace diffsol-pytorch:llvm \
  bash -lc \"maturin build --manifest-path bindings/diffsol-pytorch/Cargo.toml --release --no-default-features --features llvm,python -o dist/llvm\"
```

The resulting wheels in ``dist/llvm`` can be published or installed in other Linux environments that require Enzyme-enabled gradients.
