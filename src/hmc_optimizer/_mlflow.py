"""MLflow client wrapper for experiment tracking of HMC trials.

MLflow is also used elsewhere in the AmSC stack as an AI-gateway proxy
for LLM endpoints; this module does not touch that. See
`references/auth.md` for the narrow set of env vars relevant here.
"""
from __future__ import annotations

import os
from contextlib import contextmanager
from typing import Iterator, Mapping

from ._iri import with_retry_on_auth


def _client():
    import mlflow

    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000")
    mlflow.set_tracking_uri(tracking_uri)
    return mlflow


def experiment_name(lattice: tuple[int, int, int, int], beta: float, m_l: float) -> str:
    """Canonical experiment naming — one experiment per physics point."""
    L = "x".join(str(n) for n in lattice)
    return f"hmc_tune_{L}_{beta:g}_{m_l:g}"


@with_retry_on_auth
def ensure_experiment(name: str) -> str:
    mlflow = _client()
    exp = mlflow.get_experiment_by_name(name)
    if exp is not None:
        return exp.experiment_id
    return mlflow.create_experiment(name)


@contextmanager
def start_run(
    experiment: str,
    *,
    run_name: str | None = None,
    tags: Mapping[str, str] | None = None,
) -> Iterator[object]:
    mlflow = _client()
    exp_id = ensure_experiment(experiment)
    with mlflow.start_run(
        experiment_id=exp_id, run_name=run_name, tags=dict(tags or {})
    ) as run:
        yield run


@with_retry_on_auth
def load_history(experiment: str) -> list[dict]:
    """Return completed runs for an experiment as a list of flat dicts.

    Keys mirror the `track.py` schema: `params.*`, `metrics.*`, `tags.*`.
    """
    mlflow = _client()
    exp = mlflow.get_experiment_by_name(experiment)
    if exp is None:
        return []
    runs = mlflow.search_runs(
        experiment_ids=[exp.experiment_id],
        filter_string="attributes.status = 'FINISHED'",
        output_format="list",
    )
    out = []
    for r in runs:
        entry = {
            "run_id": r.info.run_id,
            **{f"params.{k}": v for k, v in r.data.params.items()},
            **{f"metrics.{k}": v for k, v in r.data.metrics.items()},
            **{f"tags.{k}": v for k, v in r.data.tags.items()},
        }
        out.append(entry)
    return out
