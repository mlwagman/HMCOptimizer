"""submit.build_cli_args / build_batch_script tests.

These exercise the CLI-flag surface of the precompiled Grid HMC driver
(`.claude/skills/hmc-tune/templates/driver/hmc_driver.cc`). They do not
talk to IRI, MLflow, or Grid — only to our Python glue.
"""
from __future__ import annotations

import re

import pytest

from hmc_optimizer.propose import HMCParams
from hmc_optimizer.submit import (
    RenderInputs,
    binary_for,
    build_batch_script,
    build_cli_args,
)


def _inputs(**overrides) -> RenderInputs:
    defaults = dict(
        params=HMCParams(
            integrator="omelyan", dt=0.04167, md_steps=12, traj_length=0.5
        ),
        lattice=(8, 8, 8, 8),
        beta=2.13,
        m_l=0.01,
        m_s=0.04,
        hasenbusch_ladder=[0.005, 0.0145, 0.045],
        n_trajectories=20,
    )
    defaults.update(overrides)
    return RenderInputs(**defaults)


def _pair(args: list[str], flag: str) -> str | None:
    """Return the value that follows `flag` in argv-list form."""
    for i, a in enumerate(args):
        if a == flag:
            return args[i + 1] if i + 1 < len(args) else None
    return None


def test_binary_for_maps_integrator_to_suffix():
    assert binary_for("/p/bin/hmc_driver", "leapfrog") == "/p/bin/hmc_driver_leapfrog"
    assert binary_for("/p/bin/hmc_driver", "omelyan") == "/p/bin/hmc_driver_omelyan"
    assert binary_for("/p/bin/hmc_driver", "force_gradient") == "/p/bin/hmc_driver_fg"


def test_binary_for_rejects_unknown_integrator():
    with pytest.raises(KeyError):
        binary_for("/p/bin/hmc_driver", "symplectic_4th")


def test_build_cli_args_emits_required_flags():
    args = build_cli_args(_inputs())
    assert _pair(args, "--beta") == "2.13"
    assert _pair(args, "--m-light") == "0.01"
    assert _pair(args, "--m-strange") == "0.04"
    assert _pair(args, "--md-steps") == "12"
    assert _pair(args, "--traj-length") == "0.5"


def test_build_cli_args_includes_hasenbusch_when_nonempty():
    args = build_cli_args(_inputs())
    ladder = _pair(args, "--hasenbusch")
    assert ladder is not None
    assert ladder.split(",") == ["0.005000", "0.014500", "0.045000"]
    # Rational-approx knobs come along with the ladder.
    assert _pair(args, "--rat-lo") is not None
    assert _pair(args, "--rat-degree") == "16"


def test_build_cli_args_omits_hasenbusch_when_empty():
    args = build_cli_args(_inputs(hasenbusch_ladder=[]))
    assert "--hasenbusch" not in args
    assert "--rat-lo" not in args


def test_build_cli_args_omits_m_strange_when_none():
    args = build_cli_args(_inputs(m_s=None))
    assert "--m-strange" not in args


def test_build_batch_script_wires_grid_and_mpi():
    inputs = _inputs()
    args = build_cli_args(inputs)
    script = build_batch_script(
        binary_prefix="/p/bin/hmc_driver",
        integrator="omelyan",
        cli_args=args,
        lattice=(8, 8, 8, 8),
        mpi_geom=(1, 1, 1, 4),
    )
    assert "/p/bin/hmc_driver_omelyan" in script
    assert "--grid 8.8.8.8" in script
    assert "--mpi 1.1.1.4" in script
    assert "srun -n 4 " in script
    assert re.search(r"--beta\s+2\.13", script)


def test_build_batch_script_selects_leapfrog_binary():
    inputs = _inputs(
        params=HMCParams(
            integrator="leapfrog", dt=0.04, md_steps=12, traj_length=0.5
        )
    )
    script = build_batch_script(
        binary_prefix="/p/bin/hmc_driver",
        integrator="leapfrog",
        cli_args=build_cli_args(inputs),
        lattice=(8, 8, 8, 8),
        mpi_geom=(1, 1, 1, 4),
    )
    assert "/p/bin/hmc_driver_leapfrog" in script
    assert "/p/bin/hmc_driver_omelyan" not in script


def test_build_cli_args_rejects_unknown_integrator():
    with pytest.raises(KeyError):
        build_cli_args(
            _inputs(
                params=HMCParams(
                    integrator="symplectic_4th",  # type: ignore[arg-type]
                    dt=0.04,
                    md_steps=12,
                    traj_length=0.5,
                )
            )
        )
