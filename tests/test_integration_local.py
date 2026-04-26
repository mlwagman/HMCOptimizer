"""Stage 2 integration test: exercise the HMC pipeline against a real Grid
binary, end-to-end, without IRI or Perlmutter credentials.

Skipped by default (marker `slow`). Opt in either way:

    HMC_DRIVER_PREFIX=/path/to/hmc_driver pytest -m slow tests/test_integration_local.py

The env var names the **prefix** (no `_<integrator>` suffix). The test
appends `_omelyan` to match the binary produced by
`.claude/skills/hmc-tune/templates/driver/Makefile`.

What it covers:
  1. `build_cli_args` + `run_local` actually launch the driver.
  2. The driver runs to completion (exit 0) on a trivial 4^4 HotStart run.
  3. `parse_log_file` consumes the log and yields sensible metrics.
  4. `RuleProposer.propose` accepts the resulting `TrialResult` as history.

Runtime: ~25 s on an M-series Mac; longer on slower hardware.
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest

from hmc_optimizer.parse import parse_log_file, wall_per_accept
from hmc_optimizer.propose import HMCParams, RuleProposer, TrialResult
from hmc_optimizer.submit import RenderInputs, binary_for, run_local

pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def binary_prefix() -> str:
    prefix = os.environ.get("HMC_DRIVER_PREFIX")
    if not prefix:
        pytest.skip("Set HMC_DRIVER_PREFIX to a compiled driver prefix")
    omelyan = binary_for(prefix, "omelyan")
    if not Path(omelyan).exists():
        pytest.skip(f"{omelyan} not found — build it via templates/driver/Makefile")
    return prefix


def _minimal_inputs() -> RenderInputs:
    # Physics deliberately de-tuned for a smoke test on 4^4:
    #   - HotStart so cold configs don't send the clover op singular
    #   - m_l/m_s well inside the Wilson positive-mass window
    #   - loose rat bounds since the small volume has a wide spectrum
    return RenderInputs(
        params=HMCParams(
            integrator="omelyan", dt=0.125, md_steps=4, traj_length=0.5
        ),
        lattice=(4, 4, 4, 4),
        beta=6.0,
        m_l=0.2,
        m_s=0.3,
        hasenbusch_ladder=[],
        csw=1.0,
        cg_tol=1e-6,
        cg_max=5000,
        rat_lo=1e-3,
        rat_hi=500.0,
        n_trajectories=2,
        starting_type="HotStart",
        ckpt_prefix="./ckpoint",
        ckpt_interval=10,
    )


def test_driver_runs_two_trajectories(binary_prefix: str, tmp_path: Path):
    inputs = _minimal_inputs()
    rc, log_path = run_local(
        inputs=inputs,
        binary_prefix=binary_prefix,
        run_dir=tmp_path,
        mpi_geom=(1, 1, 1, 1),
        timeout_s=300,
    )
    assert rc == 0, f"driver exited {rc}; see {log_path}"
    assert log_path.exists()

    run = parse_log_file(log_path)
    assert run.n_trajectories == 2
    # HotStart + loose tuning: we don't care about the specific dH, only that
    # it's a real finite number and acceptance isn't broken.
    assert run.accept >= 0.5
    assert run.wall_per_traj_s > 0
    assert run.plaq == run.plaq  # NaN check: parsed plaquette is finite
    wpa = wall_per_accept(run)
    assert wpa > 0 and wpa < float("inf")


def test_proposer_consumes_real_history(binary_prefix: str, tmp_path: Path):
    """A RuleProposer should accept a freshly-parsed TrialResult as history
    and return a next `HMCParams` (or `None` if already converged)."""
    inputs = _minimal_inputs()
    rc, log_path = run_local(
        inputs=inputs,
        binary_prefix=binary_prefix,
        run_dir=tmp_path,
        mpi_geom=(1, 1, 1, 1),
        timeout_s=300,
    )
    assert rc == 0

    run = parse_log_file(log_path)
    history = [
        TrialResult(
            params=inputs.params,
            accept=run.accept,
            dH=run.dH,
            plaq=run.plaq,
            wall_per_traj_s=run.wall_per_traj_s,
            wall_per_accept_s=wall_per_accept(run),
            n_trajectories=run.n_trajectories,
        )
    ]
    next_params = RuleProposer().propose(history)
    # Single-trial history never meets the stopping criterion (which needs
    # two same-integrator good trials), so we must get a concrete proposal.
    assert next_params is not None
    assert next_params.integrator == "omelyan"
