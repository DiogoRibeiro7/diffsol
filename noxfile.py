import nox


@nox.session(python=["3.9", "3.11"])
def tests(session: nox.Session) -> None:
    session.install("maturin", "pytest", "torch>=1.11")
    session.run(
        "maturin",
        "develop",
        "--manifest-path",
        "bindings/diffsol-pytorch/Cargo.toml",
        "--features",
        "python",
    )
    session.run("pytest", "bindings/diffsol-pytorch/tests")


@nox.session
def docs(session: nox.Session) -> None:
    session.install("-r", "docs/requirements.txt", "maturin")
    session.run(
        "maturin",
        "develop",
        "--manifest-path",
        "bindings/diffsol-pytorch/Cargo.toml",
        "--features",
        "python",
    )
    session.run("sphinx-build", "-b", "html", "docs", "docs/_build/html")
