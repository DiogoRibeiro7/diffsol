1. Bump the version in `bindings/diffsol-pytorch/Cargo.toml` and `pyproject.toml`.
2. Update `CHANGELOG.md`.
3. Run `scripts/build_wheel.sh` on Linux/macOS and `scripts/build_wheel.ps1` on Windows.
4. Tag the release: `git tag -a vX.Y.Z -m "Release vX.Y.Z"` and `git push --tags`.
5. The `Build Wheels` workflow publishes artifacts once the tag is pushed. Ensure `PYPI_API_TOKEN` is configured.
6. Create/update the conda-forge feedstock (see `conda/README.md`) and publish Docker images using the supplied Dockerfiles.
