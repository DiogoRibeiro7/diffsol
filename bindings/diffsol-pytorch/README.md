# diffsol-pytorch

PyTorch bindings for the [`diffsol`](https://github.com/martinjrobins/diffsol) Rust ODE solver.

## Development Workflow

```bash
python -m venv .venv
source .venv/bin/activate
pip install maturin torch
maturin develop --features python
pytest tests
```

To build wheels for release:

```bash
python -m build
```
