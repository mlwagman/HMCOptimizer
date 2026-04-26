"""OptunaProposer history-replay tests.

Skipped when Optuna isn't installed (it's an optional dependency).
"""
from __future__ import annotations

import pytest

optuna = pytest.importorskip("optuna")

from hmc_optimizer.propose import HMCParams, OptunaProposer, TrialResult


def _result(dt, md_steps, *, integrator="omelyan", accept=0.80, wpa=12.0):
    return TrialResult(
        params=HMCParams(
            integrator=integrator, dt=dt, md_steps=md_steps, traj_length=0.5
        ),
        accept=accept,
        dH=0.5,
        plaq=0.5871,
        wall_per_traj_s=wpa * accept,
        wall_per_accept_s=wpa,
        n_trajectories=20,
    )


def test_history_replay_populates_study_with_matching_integrator():
    proposer = OptunaProposer(
        integrator="omelyan", storage=None, study_name="replay_basic"
    )
    history = [
        _result(0.040, 12, accept=0.85),
        _result(0.035, 14, accept=0.80),
        _result(0.050, 10, accept=0.65),
        _result(0.045, 11, accept=0.70),
        _result(0.030, 16, accept=0.82),
    ]
    proposed = proposer.propose(history)
    assert isinstance(proposed, HMCParams)
    assert proposed.integrator == "omelyan"

    trials = proposer._study.trials
    # 5 replayed + 1 asked (not yet told) = 6
    assert len(trials) == 6
    replayed = [t for t in trials if t.user_attrs.get("replayed")]
    assert len(replayed) == 5


def test_history_replay_filters_non_matching_integrators():
    proposer = OptunaProposer(
        integrator="omelyan", storage=None, study_name="replay_filter"
    )
    history = [
        _result(0.040, 12, integrator="omelyan"),
        _result(0.040, 12, integrator="leapfrog"),
        _result(0.030, 16, integrator="force_gradient"),
    ]
    proposer.propose(history)
    replayed = [t for t in proposer._study.trials if t.user_attrs.get("replayed")]
    assert len(replayed) == 1


def test_tell_records_outcome():
    proposer = OptunaProposer(
        integrator="omelyan", storage=None, study_name="tell_basic"
    )
    proposed = proposer.propose([])
    assert proposer._last_ask is not None
    proposer.tell(
        _result(
            proposed.dt, proposed.md_steps, accept=0.80, wpa=11.0,
        )
    )
    assert proposer._last_ask is None
    # One completed trial now carries the objective value.
    completed = [t for t in proposer._study.trials if t.value is not None]
    assert len(completed) == 1


def test_objective_penalizes_out_of_band():
    in_band = _result(0.04, 12, accept=0.80, wpa=10.0)
    too_low = _result(0.04, 12, accept=0.40, wpa=10.0)
    too_high = _result(0.04, 12, accept=0.99, wpa=10.0)
    assert OptunaProposer._objective(in_band) == pytest.approx(10.0)
    assert OptunaProposer._objective(too_low) > 1e5
    assert OptunaProposer._objective(too_high) > 1e4
