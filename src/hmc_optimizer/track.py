"""MLflow logging schema for HMC trials.

The SKILL.md §7 contract: a reader should be able to `mlflow search_runs` on a
known set of param/metric/tag names and reconstruct the full tuning history.
All writers go through the `log_trial` helper so the schema stays consistent.
"""
from __future__ import annotations

import hashlib
import os
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Mapping

from . import _mlflow
from ._iri import with_retry_on_auth
from .parse import ParsedRun, wall_per_accept
from .propose import HMCParams


@dataclass(frozen=True)
class TrialContext:
    """Immutable context captured at submission time."""

    lattice: tuple[int, int, int, int]
    beta: float
    m_l: float
    m_s: float | None
    binary_path: str
    binary_sha256: str
    binary_version: str
    module_list: str
    git_sha: str
    nodes: int
    ranks_per_node: int
    machine: str
    account: str
    queue: str


def file_sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def binary_version_banner(path: str) -> str:
    try:
        out = subprocess.run(
            [path, "--version"], capture_output=True, text=True, timeout=10
        )
        return (out.stdout + out.stderr).strip()[:500]
    except Exception as e:
        return f"<unavailable: {e}>"


def current_git_sha(repo: str | Path = ".") -> str:
    try:
        out = subprocess.run(
            ["git", "-C", str(repo), "rev-parse", "HEAD"],
            capture_output=True, text=True, check=True,
        )
        return out.stdout.strip()
    except Exception:
        return "<no-git>"


def module_list_snapshot() -> str:
    ml = os.environ.get("LOADEDMODULES", "")
    return ml if ml else "<no-LOADEDMODULES>"


@with_retry_on_auth
def log_trial(
    *,
    ctx: TrialContext,
    params: HMCParams,
    parsed: ParsedRun,
    estimated_node_hours: float,
    actual_node_hours: float,
    proposer_tags: Mapping[str, str],
    artifacts: Mapping[str, str | Path] | None = None,
    phase: str = "refine",
) -> str:
    """Log one trial to MLflow and return the run_id."""
    import mlflow

    exp = _mlflow.experiment_name(ctx.lattice, ctx.beta, ctx.m_l)

    tags = {
        "phase": phase,
        "integrator": params.integrator,
        "user_email": os.environ.get("USER_EMAIL", ""),
        **dict(proposer_tags),
    }

    mlflow_params = {
        **asdict(params),
        "beta": ctx.beta,
        "m_l": ctx.m_l,
        "m_s": ctx.m_s if ctx.m_s is not None else "",
        "lattice": "x".join(str(n) for n in ctx.lattice),
        "nodes": ctx.nodes,
        "ranks_per_node": ctx.ranks_per_node,
        "machine": ctx.machine,
        "account": ctx.account,
        "queue": ctx.queue,
        "binary_path": ctx.binary_path,
        "binary_sha256": ctx.binary_sha256,
        "binary_version": ctx.binary_version,
        "module_list": ctx.module_list,
        "git_sha": ctx.git_sha,
    }

    metrics = {
        "plaq": parsed.plaq,
        "dH": parsed.dH,
        "accept": parsed.accept,
        "wall_per_traj_s": parsed.wall_per_traj_s,
        "wall_per_accept_s": wall_per_accept(parsed),
        "n_trajectories": float(parsed.n_trajectories),
        "estimated_node_hours": estimated_node_hours,
        "actual_node_hours": actual_node_hours,
    }
    if parsed.cg_iters_mean is not None:
        metrics["cg_iters_mean"] = parsed.cg_iters_mean

    with _mlflow.start_run(exp, tags=tags) as run:
        mlflow.log_params(_stringify(mlflow_params))
        for k, v in metrics.items():
            mlflow.log_metric(k, float(v))
        for name, path in (artifacts or {}).items():
            mlflow.log_artifact(str(path), artifact_path=name)
        return run.info.run_id


def _stringify(params: Mapping[str, object]) -> dict[str, str]:
    return {k: str(v) for k, v in params.items()}
