"""RuleProposer unit tests.

Each test isolates one branch of `RuleProposer.propose` by constructing a
minimal synthetic history and asserting the next `HMCParams`.
"""
from __future__ import annotations

import math

import pytest

from hmc_optimizer.propose import (
    HMCParams,
    IntractableError,
    RuleProposer,
    TrialResult,
)


def _result(
    *,
    integrator="omelyan",
    dt=0.04,
    md_steps=12,
    traj_length=0.5,
    accept=0.80,
    dH=0.5,
    wall_per_accept_s=12.5,
) -> TrialResult:
    p = HMCParams(
        integrator=integrator, dt=dt, md_steps=md_steps, traj_length=traj_length
    )
    return TrialResult(
        params=p,
        accept=accept,
        dH=dH,
        plaq=0.5871,
        wall_per_traj_s=wall_per_accept_s * accept if accept > 0 else 1.0,
        wall_per_accept_s=wall_per_accept_s,
        n_trajectories=20,
    )


def test_cold_start_matches_txqcd_recipe():
    p = RuleProposer().propose([])
    assert p.integrator == "omelyan"
    assert p.md_steps == 12
    assert p.traj_length == 0.5
    assert p.dt == pytest.approx(0.5 / 12)


def test_low_accept_halves_dt():
    hist = [_result(accept=0.30, dt=0.04)]
    p = RuleProposer().propose(hist)
    assert p.dt == pytest.approx(0.02)
    assert p.integrator == "omelyan"


def test_high_accept_grows_dt():
    hist = [_result(accept=0.98, dt=0.04)]
    p = RuleProposer().propose(hist)
    assert p.dt == pytest.approx(0.04 * 1.3)


def test_mid_accept_gentle_shrink():
    # accept in [0.6, 0.7] — below band but above the halve threshold.
    hist = [_result(accept=0.65, dt=0.04, dH=0.3)]
    p = RuleProposer().propose(hist)
    assert p.dt == pytest.approx(0.04 * 0.85)


def test_secant_step_lies_between_clamps():
    # Two same-integrator trials: accept improves as dt shrinks.
    t1 = _result(dt=0.05, md_steps=10, accept=0.60)
    t2 = _result(dt=0.04, md_steps=12, accept=0.75)
    p = RuleProposer().propose([t1, t2])
    assert p.integrator == "omelyan"
    # Secant clamped to [0.5x, 1.5x] of the most-recent dt.
    assert 0.04 * 0.5 <= p.dt <= 0.04 * 1.5


def test_secant_falls_back_when_dt_equal():
    # Equal dt → secant returns None; falls through to single-trial reaction.
    t1 = _result(dt=0.04, accept=0.30)
    t2 = _result(dt=0.04, accept=0.35)
    p = RuleProposer().propose([t1, t2])
    # Last trial accept=0.35 < 0.6 → halve dt.
    assert p.dt == pytest.approx(0.02)


def test_three_failing_leapfrog_escalates_to_omelyan():
    # Secant disabled by making the last two trials have equal dt.
    # Last trial is in-band with dH>1.0 to trigger _maybe_escalate.
    hist = [
        _result(integrator="leapfrog", dt=0.05, md_steps=10, accept=0.30, dH=2.0),
        _result(integrator="leapfrog", dt=0.05, md_steps=10, accept=0.40, dH=1.5),
        _result(integrator="leapfrog", dt=0.04, md_steps=12, accept=0.60, dH=2.0),
        _result(integrator="leapfrog", dt=0.04, md_steps=12, accept=0.75, dH=1.3),
    ]
    p = RuleProposer().propose(hist)
    assert p.integrator == "omelyan"
    assert p.dt == pytest.approx(0.04)


def test_force_gradient_gated_by_flag():
    hist = [_result(integrator="omelyan", dt=0.04, accept=0.80, dH=2.0)]
    # Default: allow_force_gradient=False → no escalation to FG.
    p_default = RuleProposer().propose(hist)
    assert p_default.integrator == "omelyan"
    # With the flag, dH>1.5 triggers escalation.
    p_allowed = RuleProposer(allow_force_gradient=True).propose(hist)
    assert p_allowed.integrator == "force_gradient"


def test_dt_floor_raises():
    # Accept below 0.6 → halve. Starting dt=0.009 halves to 0.0045 < floor 0.005.
    hist = [_result(accept=0.30, dt=0.009, md_steps=55)]
    with pytest.raises(IntractableError, match="dt"):
        RuleProposer().propose(hist)


def test_md_steps_ceiling_raises():
    # Halving dt from 0.011 → 0.0055 over traj_length=1.2 gives md_steps ≈ 218,
    # which exceeds the ceiling of 100.
    hist = [_result(accept=0.30, dt=0.011, md_steps=109, traj_length=1.2)]
    with pytest.raises(IntractableError, match="md_steps"):
        RuleProposer().propose(hist)


def test_stopping_criterion_fires_on_two_close_good_trials():
    base = dict(integrator="omelyan", dt=0.04, md_steps=12, traj_length=0.5)
    hist = [
        _result(**base, accept=0.80, dH=0.5, wall_per_accept_s=12.0),
        _result(**base, accept=0.82, dH=0.4, wall_per_accept_s=12.5),
    ]
    assert RuleProposer().propose(hist) is None


def test_stopping_criterion_does_not_fire_when_wpa_differs():
    base = dict(integrator="omelyan", dt=0.04, md_steps=12, traj_length=0.5)
    hist = [
        _result(**base, accept=0.80, dH=0.5, wall_per_accept_s=10.0),
        _result(**base, accept=0.82, dH=0.4, wall_per_accept_s=20.0),
    ]
    assert RuleProposer().propose(hist) is not None


def test_stopping_criterion_ignores_failing_trials():
    # Failing trial between good ones must not break convergence detection.
    base = dict(integrator="omelyan", dt=0.04, md_steps=12, traj_length=0.5)
    hist = [
        _result(**base, accept=0.80, dH=0.5, wall_per_accept_s=12.0),
        _result(**base, accept=0.40, dH=3.0, wall_per_accept_s=1e9),  # failing
        _result(**base, accept=0.82, dH=0.4, wall_per_accept_s=12.5),
    ]
    assert RuleProposer().propose(hist) is None
