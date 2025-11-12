Community & Maintainers
=======================

Diffsol aims to be a long-lived project shared between the scientific-computing and machine-learning communities. This page outlines communication channels, governance basics, and maintainer playbooks so contributors know where to ask questions and how decisions are made.

Communication Channels
----------------------

* **GitHub Discussions** – primary forum for Q&A, design proposals, and release announcements: https://github.com/martinjrobins/diffsol/discussions
* **Matrix / Discord bridge** – real-time chat for debugging sessions and sprints. Join via https://matrix.to/#/#diffsol:matrix.org (synced to the ``#diffsol`` Discord channel).
* **Monthly community call** – video call on the first Wednesday of each month; agenda and notes posted in Discussions.
* **Security disclosures** – email ``security@diffsol.org`` (PGP key in the repository) for vulnerabilities; we aim to triage within 48 hours.

Contribution Workflow
---------------------

1. Read :doc:`user-guide` for setup, then `CONTRIBUTING.md <../CONTRIBUTING.md>`_ for linting/tests.
2. Open an issue or discussion to scope your change.
3. Follow the pull-request template (tests, docs, benchmarks when relevant).
4. A maintainer reviews for API stability, performance, and documentation impact.

Maintainer Guidelines
---------------------

- **Triaging** – respond to new issues within 3 business days; add labels (``bug``, ``enhancement``, ``docs``) and request logs/repros if missing.
- **Review SLA** – aim to review active PRs within 5 days. If a PR is idle for >14 days, ping the author or mark as stale.
- **Release cadence** – target a MINOR release every 6–10 weeks, with PATCH releases for urgent bug fixes. See :doc:`release-strategy`.
- **Backport policy** – only security fixes and critical correctness patches are backported to the latest minor release.
- **Decision process** – consensus in Discussions; breaking changes require an RFC (template upcoming) and at least two maintainer approvals.

Community Conduct
-----------------

We expect everyone to follow the [Contributor Covenant](https://www.contributor-covenant.org/) (v2.1). Reports of harassment or unacceptable behaviour can be sent privately to the maintainers via the security contact; we will investigate promptly.

Roadmap & Working Groups
------------------------

- **Bindings** – PyTorch (current), JAX and TensorFlow prototypes planned.
- **Scientific benchmarks** – volunteers wanted for PDE/PINN testbeds.
- **Cloud & packaging** – help needed to publish wheels and container images (see :doc:`ecosystem`).

If you’d like to lead or join a working group, open a Discussion tagged ``working-group`` with your proposal.
