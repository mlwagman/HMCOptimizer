"""Parser unit tests against canned Grid-HMC log fixtures.

The fixtures mirror the message forms emitted by `Grid/qcd/hmc/HMC.h`:
    Total H after trajectory    = ...    dH = ...
    Metropolis_test -- ACCEPTED|REJECTED
    Total time for trajectory (s): ...
with optional driver-level `Plaquette:` and `ConjugateGradient converged in N iterations` lines.
"""
from __future__ import annotations

import math

import pytest

from hmc_optimizer.parse import parse_log, tail_stats, wall_per_accept


FIXTURE_MIXED = """\
Grid : Message : -- # Trajectory = 0
Grid : Message : Total H after trajectory    =   12345.670    dH =   0.032
Grid : Message : Metropolis_test -- ACCEPTED
Grid : Message : Total time for trajectory (s): 2.50
Grid : Message : Plaquette: 0.58710
Grid : Message : -- # Trajectory = 1
Grid : Message : Total H after trajectory    =   12345.700    dH =   1.210
Grid : Message : Metropolis_test -- REJECTED
Grid : Message : Total time for trajectory (s): 2.60
Grid : Message : Plaquette: 0.58700
ConjugateGradient converged in 120 iterations
Grid : Message : -- # Trajectory = 2
Grid : Message : Total H after trajectory    =   12345.680    dH =   0.150
Grid : Message : Metropolis_test -- ACCEPTED
Grid : Message : Total time for trajectory (s): 2.55
Grid : Message : Plaquette: 0.58720
ConjugateGradient converged in 118 iterations
"""

FIXTURE_ALL_ACCEPTED = """\
Total H after trajectory = 1.00 dH = 0.01
Metropolis_test -- ACCEPTED
Total time for trajectory (s): 1.00
Total H after trajectory = 2.00 dH = 0.02
Metropolis_test -- ACCEPTED
Total time for trajectory (s): 1.20
"""

FIXTURE_ALL_REJECTED = """\
Total H after trajectory = 1.00 dH = 5.00
Metropolis_test -- REJECTED
Total time for trajectory (s): 1.00
Total H after trajectory = 2.00 dH = 6.00
Metropolis_test -- REJECTED
Total time for trajectory (s): 1.10
"""

FIXTURE_NO_DRIVER_EXTRAS = """\
Total H after trajectory = 1.00 dH = 0.50
Metropolis_test -- ACCEPTED
Total time for trajectory (s): 1.00
"""


def test_parse_mixed():
    r = parse_log(FIXTURE_MIXED)
    assert r.n_trajectories == 3
    assert r.accept == pytest.approx(2 / 3)
    assert r.dH == pytest.approx((0.032 + 1.210 + 0.150) / 3)
    assert r.plaq == pytest.approx((0.5871 + 0.5870 + 0.5872) / 3, rel=1e-4)
    assert r.wall_per_traj_s == pytest.approx((2.50 + 2.60 + 2.55) / 3)
    assert r.cg_iters_mean == pytest.approx((120 + 118) / 2)


def test_parse_all_accepted():
    r = parse_log(FIXTURE_ALL_ACCEPTED)
    assert r.accept == 1.0
    assert r.n_trajectories == 2
    assert wall_per_accept(r) == pytest.approx(1.10)


def test_parse_all_rejected():
    r = parse_log(FIXTURE_ALL_REJECTED)
    assert r.accept == 0.0
    assert r.n_trajectories == 2
    assert math.isinf(wall_per_accept(r))


def test_parse_missing_metropolis_raises():
    with pytest.raises(ValueError, match="Metropolis_test"):
        parse_log("some arbitrary output with no accept/reject lines")


def test_parse_without_driver_extras():
    r = parse_log(FIXTURE_NO_DRIVER_EXTRAS)
    assert r.n_trajectories == 1
    assert r.accept == 1.0
    assert math.isnan(r.plaq)
    assert r.cg_iters_mean is None
    assert r.raw_sample_count == 0


def test_tail_stats_trims_thermalization():
    # First 10 trajectories all accept; next 10 all reject. Tail=10 should report 0% accept.
    lines = []
    for i in range(10):
        lines += [
            f"Total H after trajectory = {i}.0 dH = 0.01",
            "Metropolis_test -- ACCEPTED",
            "Total time for trajectory (s): 1.00",
        ]
    for i in range(10, 20):
        lines += [
            f"Total H after trajectory = {i}.0 dH = 5.00",
            "Metropolis_test -- REJECTED",
            "Total time for trajectory (s): 1.00",
        ]
    text = "\n".join(lines)

    full = parse_log(text)
    assert full.n_trajectories == 20
    assert full.accept == pytest.approx(0.5)

    tailed = tail_stats([text], tail=10)
    assert tailed.n_trajectories == 10
    assert tailed.accept == 0.0
