# HMCOptimizer

Adaptive HMC hyperparameter tuning for pure-QCD Grid binaries, driven through
the AmSC stack: IRI for job submission, MLflow for experiment tracking, Globus
for checkpoint staging.

Sibling to [`HadronsJobBuilder`](../HadronsJobBuilder) — this package covers
gauge generation, HadronsJobBuilder covers measurement. The only point of
coupling is `src/hmc_optimizer/_iri.py`, which re-exports the narrow IRI
surface this driver needs.

## Quickstart

```bash
pip install -e ../HadronsJobBuilder
pip install -e .[optuna]

mlflow server --port 5000 &        # local tracking server
```

Open Claude Code here; the `hmc-tune` skill (under `.claude/skills/`) auto-loads
and an agent can walk the decision flow from a single prompt, e.g.

> "Tune HMC for an 8⁴ pure-QCD ensemble on Perlmutter at β=2.13, m_l=0.01.
> Binary at `/global/.../hmc_pureqcd`."

The skill calls `scripts/hmc_submit.py`, then `scripts/hmc_status.py`, and
loops via `scripts/hmc_report.py` until the stopping criterion fires
(acceptance ∈ [0.7, 0.9], dH < 1.0, wall/accept stable within 10%).

## Files

- `.claude/skills/hmc-tune/` — the skill agents load: `SKILL.md`, reference
  deep-dives, and three Jinja templates for the Grid HMC driver.
- `src/hmc_optimizer/` — Python package. Keep new cross-project imports
  isolated to `_iri.py`.
- `scripts/` — CLI entry points `hmc_submit`, `hmc_status`, `hmc_report`.

## Cost discipline

The skill refuses to submit above a cumulative 100 node-hour cap without
`--node-hours-cap` confirmation. Every trial logs both estimated and actual
node-hours so miscalibrations are visible.
