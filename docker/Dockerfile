FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends build-essential curl && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace
COPY . /workspace

RUN pip install --upgrade pip maturin torch \
    && maturin develop --locked --manifest-path bindings/diffsol-pytorch/Cargo.toml --features python \
    && pip install -r docs/requirements.txt

CMD ["python"]
