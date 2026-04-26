"""Build CLI args + batch script for a precompiled Grid HMC binary.

HMCOptimizer ships one .cc source (`.claude/skills/hmc-tune/templates/driver/
hmc_driver.cc`) that is built three times on Perlmutter (one per integrator)
via the accompanying Makefile. Every hyperparameter we tune is a runtime CLI
flag, so submission at trial time never touches source — it just picks the
right binary and composes argv.

Flow:

    build_cli_args(inputs)  → ["--beta", "6.1", "--md-steps", "10", ...]
    build_batch_script(...) → "srun -n N <binary> <args> > hmc_run.log"
    submit_trial(...)       → IRI job id
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from .propose import HMCParams

# Mapping from our integrator name to the binary suffix produced by the
# driver Makefile. The Makefile builds `hmc_driver_<suffix>` for each.
_BINARY_SUFFIX: dict[str, str] = {
    "leapfrog": "leapfrog",
    "omelyan": "omelyan",
    "force_gradient": "fg",
}


@dataclass(frozen=True)
class RenderInputs:
    params: HMCParams
    lattice: tuple[int, int, int, int]
    beta: float
    m_l: float
    m_s: float | None
    hasenbusch_ladder: Sequence[float]
    u0: float = 0.8326053
    csw: float = 1.0
    stout_rho: float = 0.125
    stout_nsmear: int = 1
    hasenbusch_top: float = 1.0
    rat_lo: float = 1.0e-4
    rat_hi: float = 200.0
    rat_degree: int = 16
    rat_precision: int = 64
    cg_tol: float = 1.0e-8
    cg_max: int = 30000
    n_trajectories: int = 20
    no_metropolis_until: int = 0
    start_trajectory: int = 0
    starting_type: str = "ColdStart"
    ckpt_prefix: str = "ckpoint"
    ckpt_interval: int = 5


def binary_for(binary_prefix: str, integrator: str) -> str:
    """Return the concrete binary path for the requested integrator.

    `binary_prefix` is the absolute path up to (but excluding) the
    `_<suffix>` tail, e.g. `/global/.../bin/hmc_driver`.
    """
    if integrator not in _BINARY_SUFFIX:
        raise KeyError(f"Unknown integrator: {integrator!r}")
    return f"{binary_prefix}_{_BINARY_SUFFIX[integrator]}"


def build_cli_args(inputs: RenderInputs) -> list[str]:
    """Return the argv list passed to the driver binary (excluding argv[0])."""
    if inputs.params.integrator not in _BINARY_SUFFIX:
        raise KeyError(f"Unknown integrator: {inputs.params.integrator!r}")

    args: list[str] = [
        "--beta", f"{inputs.beta}",
        "--u0", f"{inputs.u0}",
        "--m-light", f"{inputs.m_l}",
        "--csw", f"{inputs.csw}",
        "--stout-rho", f"{inputs.stout_rho}",
        "--stout-nsmear", f"{inputs.stout_nsmear}",
        "--md-steps", f"{inputs.params.md_steps}",
        "--traj-length", f"{inputs.params.traj_length}",
        "--cg-tol", f"{inputs.cg_tol}",
        "--cg-max", f"{inputs.cg_max}",
        "--n-trajectories", f"{inputs.n_trajectories}",
        "--no-metropolis-until", f"{inputs.no_metropolis_until}",
        "--start-trajectory", f"{inputs.start_trajectory}",
        "--starting-type", inputs.starting_type,
        "--ckpt-prefix", inputs.ckpt_prefix,
        "--ckpt-interval", f"{inputs.ckpt_interval}",
    ]
    if inputs.m_s is not None:
        args += ["--m-strange", f"{inputs.m_s}"]
    if inputs.hasenbusch_ladder:
        args += [
            "--hasenbusch",
            ",".join(f"{m:.6f}" for m in inputs.hasenbusch_ladder),
            "--hasenbusch-top", f"{inputs.hasenbusch_top}",
            "--rat-lo", f"{inputs.rat_lo}",
            "--rat-hi", f"{inputs.rat_hi}",
            "--rat-degree", f"{inputs.rat_degree}",
            "--rat-precision", f"{inputs.rat_precision}",
        ]
    return args


def build_batch_script(
    *,
    binary_prefix: str,
    integrator: str,
    cli_args: Sequence[str],
    lattice: tuple[int, int, int, int],
    mpi_geom: tuple[int, int, int, int],
    log_basename: str = "hmc_run",
) -> str:
    """Compose a bash batch script that invokes the driver under srun."""
    nranks = mpi_geom[0] * mpi_geom[1] * mpi_geom[2] * mpi_geom[3]
    mpi_str = ".".join(str(n) for n in mpi_geom)
    grid_str = ".".join(str(n) for n in lattice)
    binary = binary_for(binary_prefix, integrator)
    cli = " ".join(cli_args)
    return (
        "set -e\n"
        f"srun -n {nranks} {binary} "
        f"--grid {grid_str} --mpi {mpi_str} "
        f"{cli} "
        f"> {log_basename}.log 2>&1\n"
    )


def estimate_node_hours(
    *,
    md_steps: int,
    n_trajectories: int,
    nodes: int,
    est_sec_per_md_step: float = 3.0,
) -> float:
    total_seconds = md_steps * n_trajectories * est_sec_per_md_step
    return (nodes * total_seconds) / 3600.0


def run_local(
    *,
    inputs: RenderInputs,
    binary_prefix: str,
    run_dir: str | Path,
    mpi_geom: tuple[int, int, int, int] = (1, 1, 1, 1),
    log_basename: str = "hmc_run",
    mpi_launcher: Sequence[str] | None = None,
    extra_env: dict[str, str] | None = None,
    timeout_s: float | None = None,
) -> tuple[int, Path]:
    """Execute the HMC driver via subprocess in `run_dir`.

    Stage 2 integration testing: same argv construction as `submit_trial`,
    but runs locally instead of shipping to IRI. Returns `(returncode,
    log_path)`. The caller is responsible for parsing the log afterwards.

    `mpi_launcher` prepends e.g. `["mpirun", "-n", "4"]`. When omitted, the
    binary is invoked directly (single rank) — only valid for mpi_geom
    (1,1,1,1).
    """
    import os
    import subprocess

    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / f"{log_basename}.log"

    nranks = mpi_geom[0] * mpi_geom[1] * mpi_geom[2] * mpi_geom[3]
    if mpi_launcher is None and nranks != 1:
        raise ValueError(
            f"mpi_launcher required for multi-rank geom {mpi_geom}; "
            "pass e.g. ['mpirun', '-n', '4']."
        )

    binary = binary_for(binary_prefix, inputs.params.integrator)
    grid_str = ".".join(str(n) for n in inputs.lattice)
    mpi_str = ".".join(str(n) for n in mpi_geom)

    argv: list[str] = list(mpi_launcher or [])
    argv += [binary, "--grid", grid_str, "--mpi", mpi_str]
    argv += build_cli_args(inputs)

    env = os.environ.copy()
    if extra_env:
        env.update(extra_env)

    with open(log_path, "wb") as log_file:
        result = subprocess.run(
            argv,
            cwd=str(run_dir),
            env=env,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            timeout=timeout_s,
            check=False,
        )
    return result.returncode, log_path


def submit_trial(
    *,
    machine: str,
    account: str,
    queue: str,
    time_seconds: int,
    nodes: int,
    ranks_per_node: int,
    gpus_per_rank: int,
    inputs: RenderInputs,
    binary_prefix: str,
    mpi_geom: tuple[int, int, int, int],
    run_dir: str,
    dry_run: bool = False,
) -> str | None:
    """Submit a single HMC trial.

    `binary_prefix` is the path up to (but excluding) the `_<suffix>` tail;
    the integrator in `inputs.params` picks the concrete binary.
    """
    script_body = build_batch_script(
        binary_prefix=binary_prefix,
        integrator=inputs.params.integrator,
        cli_args=build_cli_args(inputs),
        lattice=inputs.lattice,
        mpi_geom=mpi_geom,
    )

    if dry_run:
        print("=== dry-run batch script ===")
        print(script_body)
        return None

    from . import _iri

    _iri.remote_mkdir(machine, run_dir, create_parents=True)
    return _iri.submit_batch_job(
        machine,
        script_body,
        nodes=nodes,
        ranks_per_node=ranks_per_node,
        gpus_per_rank=gpus_per_rank,
        time=str(time_seconds),
        queue=queue,
        account=account,
        job_run_dir=run_dir,
        exclusive=False,
        allow_unsafe=False,
    )
